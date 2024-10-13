import os
join = os.path.join
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import monai
from segment_anything import sam_model_registry
import argparse
from datetime import datetime
import shutil
from metrics import SegMetrics
from post_process import RefinementNet
from utils.cldice import soft_cldice
from datasets import VesselDataset
from torch.optim.lr_scheduler import ExponentialLR
from net import SAM, RefinementNet

torch.manual_seed(114514)
torch.cuda.empty_cache()

# Set up Parser
parser = argparse.ArgumentParser()
parser.add_argument("-task_name", type=str, default="UniVesselSeg")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument("-checkpoint", type=str, default="../sam_vit_b_01ec64.pth")
parser.add_argument("-work_dir", type=str, default="./workdir")

# Train
parser.add_argument("-num_epochs_sam", type=int, default=20)
parser.add_argument("-num_epochs_refine", type=int, default=5)
parser.add_argument("-batch_size", type=int, default=2)
parser.add_argument("-num_workers", type=int, default=4)
parser.add_argument("--train_dataset", type=str, default="../train")
parser.add_argument("--device", type=str, default="cuda:0")

# Optimizer Parameters
parser.add_argument("-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)")
parser.add_argument("-lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)")
parser.add_argument("--resume", type=str, default="", help="Resuming training from checkpoint")
parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
args = parser.parse_args()


run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)


def main():
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )
    loss_log_path = os.path.join(model_save_path, "loss_log.txt")
    with open(loss_log_path, "a") as f:
        f.write("Epoch\tLR\tTrain Loss\tVal Loss\n")
    
    # Define SAM and Refinement Net
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam_model = SAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    refine_net = RefinementNet().to(device)

    # Freeze SAM Vit Encoder
    exclude_keywords = ['Adapter', 'convnext', 'FPN', 'crossattnfusion']
    for name, param in sam_model.image_encoder.named_parameters():
        if not any(keyword in name for keyword in exclude_keywords):
            param.requires_grad = False
    
    print("Number of Sam Parameters(Including Freeze Parameters): ", sum(p.numel() for p in sam_model.parameters()),)
    print(" Number of Sam Trainable Parameters: ",sum(p.numel() for p in sam_model.parameters() if p.requires_grad),)
    print("Number of Refinement Net parameters:", sum(p.numel() for p in refine_net.parameters()),)
    print('workdir:', args.work_dir)

    # Define Optimizer
    optimizer_sam = torch.optim.AdamW(sam_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_refinement = torch.optim.AdamW(refine_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ExponentialLR(optimizer_refinement, gamma=0.95)

    # Define SAM Loss
    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")

    # Define Refinement Net Loss
    cl_dice = soft_cldice()
    mse_loss = nn.MSELoss(reduction="mean")

    num_epochs_sam = args.num_epochs_sam
    num_epochs_refine = args.num_epochs_refine

    train_losses = []
    val_losses = []
    best_loss = 1e10
    train_dataset = VesselDataset(args.train_dataset, mode='train')
    val_dataset = VesselDataset(args.train_dataset, mode='val')

    print("Number of training samples: ", len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    start_epoch = 0

    if args.resume is not None:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            sam_model.load_state_dict(checkpoint["model"])
            optimizer_sam.load_state_dict(checkpoint["optimizer"])
        
    l = len(train_dataloader)
    l_val = len(val_dataloader)

    # SAM Training/Validation
    print("*******Training SAM Stage*******")
    for epoch in range(start_epoch, num_epochs_sam):
        val_iter_metrics = [0] * len(args.metrics)
        train_iter_metrics = [0] * len(args.metrics)
        train_epoch_loss = 0
        val_epoch_loss = 0

        sam_model.train()
        for step_train, (image, gt2D, point_coords, point_labels, _) in enumerate(tqdm(train_dataloader)):
            optimizer_sam.zero_grad()
            point_coords = point_coords.detach().cpu().numpy()
            point_labels = point_labels.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)

            sam_pred = sam_model(image, point_coords, point_labels)
            loss = seg_loss(sam_pred, gt2D) + ce_loss(sam_pred, gt2D.float())
            loss.backward()
            optimizer_sam.step()
            optimizer_sam.zero_grad()
            
            train_batch_metrics = SegMetrics(gt2D, sam_pred, args.metrics)
            train_iter_metrics = [train_iter_metrics[i] + train_batch_metrics[i] for i in range(len(args.metrics))]
            train_epoch_loss += loss.item()

        sam_model.eval()
        with torch.no_grad():
            for step_val, (image, gt2D, point_coords, point_labels, _) in enumerate(tqdm(val_dataloader)):
                point_coords = point_coords.detach().cpu().numpy()
                point_labels = point_labels.detach().cpu().numpy()
                image, gt2D = image.to(device), gt2D.to(device)

                sam_pred = sam_model(image, point_coords, point_labels)
                loss = seg_loss(sam_pred, gt2D) + ce_loss(sam_pred, gt2D.float())

                val_epoch_loss += loss.item()
                val_batch_metrics = SegMetrics(gt2D, sam_pred, args.metrics)
                val_iter_metrics = [val_iter_metrics[i] + val_batch_metrics[i] for i in range(len(args.metrics))]
        
        train_iter_metrics = [metric / l for metric in train_iter_metrics]
        train_metrics = {args.metrics[i]: '{:.4f}'.format(train_iter_metrics[i]) for i in range(len(train_iter_metrics))}

        val_iter_metrics = [metric / l_val for metric in val_iter_metrics]
        val_metrics = {args.metrics[i]: '{:.4f}'.format(val_iter_metrics[i]) for i in range(len(val_iter_metrics))}

        train_epoch_loss /= step_train
        val_epoch_loss /= step_val
        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)
        with open(loss_log_path, "a") as f:
            f.write(f"{epoch}\t{args.lr}\t{train_epoch_loss:.6f}\t{val_epoch_loss:.6f}\n")
        print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Train_loss: {train_epoch_loss}, Train metrics: {train_metrics}, Val_loss:{val_epoch_loss}, Val metrics: {val_metrics}')
        
        # Save the Latest SAM Model
        checkpoint = {
            "model": sam_model.state_dict(),
            "optimizer": optimizer_sam.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_path, "sam_model_latest.pth"))

        ## Save the Best SAM Model
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            checkpoint = {
                "model": sam_model.state_dict(),
                "optimizer": optimizer_sam.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "sam_model_best.pth"))
    
    torch.cuda.empty_cache()
    print("*******Training Refinement Net Stage*******")

    # Load Best Sam pth
    sam_model = sam_model_registry[args.model_type](checkpoint=join(model_save_path, "sam_model_best.pth"))
    sam_model = SAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)

    sam_model.eval()
    train_dataset_refine = VesselDataset(args.train_dataset, mode='train')
    val_dataset_refine = VesselDataset(args.train_dataset, mode='val')

    print("Number of training samples: ", len(train_dataset))
    train_dataloader_re = DataLoader(
        train_dataset_refine,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_dataloader_re = DataLoader(
        val_dataset_refine,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    best_loss = 1e4

    with open(loss_log_path, "a") as f:
        f.write("**Start Training Refinement Net**\n")
    l = len(train_dataloader_re)
    l_val = len(val_dataloader_re)

    for epoch in range(start_epoch, num_epochs_refine):
        val_iter_metrics = [0] * len(args.metrics)
        train_iter_metrics = [0] * len(args.metrics)
        train_epoch_loss = 0
        val_epoch_loss = 0
        step_train = 0
        step_val = 0

        refine_net.train()
        for step_train, (image, gt2D, point_coords, point_labels, _) in enumerate(tqdm(train_dataloader_re)):
            optimizer_refinement.zero_grad()
            point_coords = point_coords.detach().cpu().numpy()
            point_labels = point_labels.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)

            with torch.no_grad():
                mask = sam_model(image, point_coords, point_labels)

            mask_refine = refine_net(mask)
            
            loss_refine = mse_loss(mask_refine, gt2D.float()) + cl_dice(gt2D.float(), mask_refine)
            loss_refine.backward()

            optimizer_refinement.step()
            optimizer_refinement.zero_grad()
            
            train_batch_metrics = SegMetrics(gt2D, mask_refine, args.metrics)
            train_iter_metrics = [train_iter_metrics[i] + train_batch_metrics[i] for i in range(len(args.metrics))]
            train_epoch_loss += loss_refine.item()

        refine_net.eval()
        with torch.no_grad():
            for step_val, (image, gt2D, point_coords, point_labels, _) in enumerate(tqdm(val_dataloader_re)):
                point_coords = point_coords.detach().cpu().numpy()
                point_labels = point_labels.detach().cpu().numpy()
                image, gt2D = image.to(device), gt2D.to(device)

                mask = sam_model(image, point_coords, point_labels)
                mask_refine = refine_net(mask)

                loss_refine = mse_loss(mask_refine, gt2D.float()) + cl_dice(gt2D.float(), mask_refine)

                val_epoch_loss += loss_refine.item()
                val_batch_metrics = SegMetrics(gt2D, mask_refine, args.metrics)
                val_iter_metrics = [val_iter_metrics[i] + val_batch_metrics[i] for i in range(len(args.metrics))]
        
        
        train_iter_metrics = [metric / l for metric in train_iter_metrics]
        train_metrics = {args.metrics[i]: '{:.4f}'.format(train_iter_metrics[i]) for i in range(len(train_iter_metrics))}

        val_iter_metrics = [metric / l_val for metric in val_iter_metrics]
        val_metrics = {args.metrics[i]: '{:.4f}'.format(val_iter_metrics[i]) for i in range(len(val_iter_metrics))}
        train_epoch_loss /= step_train
        val_epoch_loss /= step_val
        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)

        scheduler.step()
        with open(loss_log_path, "a") as f:
            f.write(f"{epoch}\t{args.lr}\t{train_epoch_loss:.6f}\t{val_epoch_loss:.6f}\n")
        print(f'Time: {datetime.now().strftime("%Y%m%d-%H%M")}, Epoch: {epoch}, Train_loss: {train_epoch_loss}, Train metrics: {train_metrics}, Val_loss:{val_epoch_loss}, Val metrics: {val_metrics}')
        # Save the Latest Refinement Net
        checkpoint = {
            "model": refine_net.state_dict(),
            "optimizer": optimizer_refinement.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint, join(model_save_path, "refinement_model_latest.pth"))
        # Save the Latest Best Refinement Net
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            checkpoint = {
                "model": refine_net.state_dict(),
                "optimizer": optimizer_refinement.state_dict(),
                "epoch": epoch,
            }
            torch.save(checkpoint, join(model_save_path, "refinement_model_best.pth"))


if __name__ == "__main__":
    main()
