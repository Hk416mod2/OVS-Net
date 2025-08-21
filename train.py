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
from utils.cldice import soft_cldice
from utils.logging import setup_logging, log_epoch_results, log_training_summary, log_final_summary, print_training_info
from datasets import VesselDataset
from torch.optim.lr_scheduler import ExponentialLR
from net import SAM, RefinementNet
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

torch.manual_seed(2025)
torch.cuda.empty_cache()


parser = argparse.ArgumentParser()
parser.add_argument("-task_name", type=str, default="VesselSeg")
parser.add_argument("-model_type", type=str, default="vit_b")
parser.add_argument("-checkpoint", type=str, default="/mnt/siat225_disk1/SDN/OVS-Net/UniVesselSeg/SAM-Med2D/pretrain_model/sam_vit_b_01ec64.pth")
parser.add_argument("-work_dir", type=str, default="workdir")

parser.add_argument("-num_epochs_sam", type=int, default=20)
parser.add_argument("-num_epochs_refine", type=int, default=10)
parser.add_argument("-batch_size", type=int, default=2)
parser.add_argument("-num_workers", type=int, default=4)
parser.add_argument("--train_dataset", type=str, default="./Dataset/train")
parser.add_argument("--device", type=str, default="cuda:0")

parser.add_argument("-weight_decay", type=float, default=0.01, help="weight decay (default: 0.01)")
parser.add_argument("-lr", type=float, default=0.0001, metavar="LR", help="learning rate (absolute lr)")
parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
parser.add_argument("--use_amp", action="store_true", default=True, help="Enable automatic mixed precision")
parser.add_argument("--amp_scale", type=float, default=2**16, help="Initial scale for GradScaler")
args = parser.parse_args()


run_id = datetime.now().strftime("%Y%m%d-%H%M")
model_save_path = join(args.work_dir, args.task_name + "-" + run_id)
device = torch.device(args.device)


def main():
    os.makedirs(model_save_path, exist_ok=True)
    shutil.copyfile(
        __file__, join(model_save_path, run_id + "_" + os.path.basename(__file__))
    )
    
    train_log_path, sam_log_path, refine_log_path = setup_logging(model_save_path, run_id, args)
    
    # Initialize AMP Training
    if args.use_amp:
        scaler_sam = GradScaler(init_scale=args.amp_scale)
        scaler_refine = GradScaler(init_scale=args.amp_scale)
        print(f"using amp: {args.amp_scale}")
    else:
        scaler_sam = None
        scaler_refine = None
        print("using standard precision training")
    
    sam_model = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
    sam_model = SAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    refine_net = RefinementNet().to(device)

    exclude_keywords = ['Adapter', 'convnext', 'FPN', 'crossattnfusion', '11']
    for name, param in sam_model.image_encoder.named_parameters():
        if not any(keyword in name for keyword in exclude_keywords):
            param.requires_grad = False
    
    print("Number of Seg Model Parameters(Including Frozen Parameters): ", sum(p.numel() for p in sam_model.parameters()),)
    print("Number of Seg Model Trainable Parameters: ",sum(p.numel() for p in sam_model.parameters() if p.requires_grad),)
    print("Number of Refinement Net parameters:", sum(p.numel() for p in refine_net.parameters()),)
    print('workdir:', args.work_dir)

    optimizer_sam = torch.optim.AdamW(sam_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_refinement = torch.optim.AdamW(refine_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ExponentialLR(optimizer_refinement, gamma=0.95)

    seg_loss = monai.losses.DiceLoss(sigmoid=True, squared_pred=True, reduction="mean")
    ce_loss = nn.BCEWithLogitsLoss(reduction="mean")
    cl_dice = soft_cldice()
    mse_loss = nn.MSELoss(reduction="mean")

    train_dataset = VesselDataset(args.train_dataset, mode='train')
    val_dataset = VesselDataset(args.train_dataset, mode='val')

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


    print("*******Training SAM Model Stage with AMP*******")
    best_sam_metrics = train_sam(sam_model, train_dataloader, val_dataloader, optimizer_sam,
                                    seg_loss, ce_loss, args.num_epochs_sam, model_save_path, 
                                    args, sam_log_path, scaler_sam)

    log_training_summary(train_log_path, "SAM Model (AMP)", best_sam_metrics, args.num_epochs_sam)

    torch.cuda.empty_cache()
    print("*******Training Refinement Net Stage with AMP*******")
    
    sam_model = sam_model_registry[args.model_type](checkpoint=join(model_save_path, "sam_model_best.pth"))
    sam_model = SAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)
    sam_model.eval()

    best_refine_metrics = train_refine_net(refine_net, sam_model, train_dataloader, val_dataloader, 
                                              optimizer_refinement, scheduler, mse_loss, cl_dice, 
                                              args.num_epochs_refine, model_save_path, args, refine_log_path, scaler_refine)

    log_training_summary(train_log_path, "Refinement Network (AMP)", best_refine_metrics, args.num_epochs_refine)
    log_final_summary(train_log_path, model_save_path)


def train_sam(model, train_loader, val_loader, optimizer, seg_loss, ce_loss, num_epochs, save_path, args, log_path, scaler):
    """AMP SAM Model Training"""
    best_val_metrics = {metric: 0.0 for metric in args.metrics}
    patience_counter = 0
    patience = 5
    l = len(train_loader)
    l_val = len(val_loader)
    
    for epoch in range(num_epochs):
        val_iter_metrics = [0] * len(args.metrics)
        train_iter_metrics = [0] * len(args.metrics)
        train_epoch_loss = 0
        val_epoch_loss = 0

        model.train()
        for step_train, (image, gt2D, point_coords, point_labels, _) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            point_coords = point_coords.detach().cpu().numpy()
            point_labels = point_labels.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)

            # Use AMP Training
            if scaler is not None:
                with autocast():
                    sam_pred = model(image, point_coords, point_labels)
                    loss = 0.6 * seg_loss(sam_pred, gt2D) + 0.4 * ce_loss(sam_pred, gt2D.float())
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                sam_pred = model(image, point_coords, point_labels)
                loss = 0.6 * seg_loss(sam_pred, gt2D) + 0.4 * ce_loss(sam_pred, gt2D.float())
                loss.backward()
                optimizer.step()
            
            train_batch_metrics = SegMetrics(gt2D, sam_pred, args.metrics)
            train_iter_metrics = [train_iter_metrics[i] + train_batch_metrics[i] for i in range(len(args.metrics))]
            train_epoch_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for step_val, (image, gt2D, point_coords, point_labels, _) in enumerate(tqdm(val_loader)):
                point_coords = point_coords.detach().cpu().numpy()
                point_labels = point_labels.detach().cpu().numpy()
                image, gt2D = image.to(device), gt2D.to(device)

                if scaler is not None:
                    with autocast():
                        sam_pred = model(image, point_coords, point_labels)
                        loss = 0.6 * seg_loss(sam_pred, gt2D) + 0.4 * ce_loss(sam_pred, gt2D.float())
                else:
                    sam_pred = model(image, point_coords, point_labels)
                    loss = 0.6 * seg_loss(sam_pred, gt2D) + 0.4 * ce_loss(sam_pred, gt2D.float())

                val_epoch_loss += loss.item()
                val_batch_metrics = SegMetrics(gt2D, sam_pred, args.metrics)
                val_iter_metrics = [val_iter_metrics[i] + val_batch_metrics[i] for i in range(len(args.metrics))]
        
        train_iter_metrics = [metric / l for metric in train_iter_metrics]
        train_metrics = {args.metrics[i]: train_iter_metrics[i] for i in range(len(train_iter_metrics))}

        val_iter_metrics = [metric / l_val for metric in val_iter_metrics]
        val_metrics = {args.metrics[i]: val_iter_metrics[i] for i in range(len(val_iter_metrics))}

        train_epoch_loss /= (step_train + 1)
        val_epoch_loss /= (step_val + 1)

        amp_info = ""
        if scaler is not None:
            amp_info = f" | AMP Scale: {scaler.get_scale():.0f}"

        log_epoch_results(log_path, epoch, args.lr, train_epoch_loss, val_epoch_loss, train_metrics, val_metrics, args)
        
        print_training_info(epoch, train_epoch_loss, val_epoch_loss, train_metrics, val_metrics)
        if amp_info:
            print(f"AMP Info: {amp_info}")
        
        current_val_metrics = {args.metrics[i]: float(val_iter_metrics[i]) / l_val for i in range(len(args.metrics))}
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        if scaler is not None:
            checkpoint["scaler"] = scaler.state_dict()
        
        torch.save(checkpoint, join(save_path, "sam_model_latest.pth"))
        
        improved = False
        for metric in args.metrics:
            if current_val_metrics[metric] > best_val_metrics[metric]:
                best_val_metrics[metric] = current_val_metrics[metric]
                improved = True
        
        if improved:
            patience_counter = 0
            torch.save(checkpoint, join(save_path, "sam_model_best.pth"))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    return best_val_metrics


def train_refine_net(refine_net, sam_model, train_loader, val_loader, optimizer, scheduler, 
                         mse_loss, cl_dice, num_epochs, save_path, args, log_path, scaler):
    """AMP Refinement Network Training"""
    best_loss = 1e4
    best_val_metrics = {metric: 0.0 for metric in args.metrics}
    l = len(train_loader)
    l_val = len(val_loader)

    for epoch in range(num_epochs):
        val_iter_metrics = [0] * len(args.metrics)
        train_iter_metrics = [0] * len(args.metrics)
        train_epoch_loss = 0
        val_epoch_loss = 0
        step_train = 0
        step_val = 0

        refine_net.train()
        for step_train, (image, gt2D, point_coords, point_labels, _) in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            point_coords = point_coords.detach().cpu().numpy()
            point_labels = point_labels.detach().cpu().numpy()
            image, gt2D = image.to(device), gt2D.to(device)

            with torch.no_grad():
                if scaler is not None:
                    with autocast():
                        sam_pred = sam_model(image, point_coords, point_labels)
                else:
                    sam_pred = sam_model(image, point_coords, point_labels)
            
            sam_pred_512 = F.interpolate(sam_pred, size=(512, 512), mode='bilinear', align_corners=False)
            
            if scaler is not None:
                with autocast():
                    refined_pred_512 = refine_net(sam_pred_512)
                    refined_pred = F.interpolate(refined_pred_512, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=False)
                    loss = 0.4 * mse_loss(refined_pred, gt2D.float()) + 0.6 * cl_dice(gt2D.float(), refined_pred)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                refined_pred_512 = refine_net(sam_pred_512)
                refined_pred = F.interpolate(refined_pred_512, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=False)
                loss = 0.4 * mse_loss(refined_pred, gt2D.float()) + 0.6 * cl_dice(gt2D.float(), refined_pred)
                loss.backward()
                optimizer.step()
            
            train_batch_metrics = SegMetrics(gt2D, refined_pred, args.metrics)
            train_iter_metrics = [train_iter_metrics[i] + train_batch_metrics[i] for i in range(len(args.metrics))]
            train_epoch_loss += loss.item()

        refine_net.eval()
        with torch.no_grad():
            for step_val, (image, gt2D, point_coords, point_labels, _) in enumerate(tqdm(val_loader)):
                point_coords = point_coords.detach().cpu().numpy()
                point_labels = point_labels.detach().cpu().numpy()
                image, gt2D = image.to(device), gt2D.to(device)

                if scaler is not None:
                    with autocast():
                        sam_pred = sam_model(image, point_coords, point_labels)
                        sam_pred_512 = F.interpolate(sam_pred, size=(512, 512), mode='bilinear', align_corners=False)
                        refined_pred_512 = refine_net(sam_pred_512)
                        refined_pred = F.interpolate(refined_pred_512, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=False)
                        loss = 0.4 * mse_loss(refined_pred, gt2D.float()) + 0.6 * cl_dice(gt2D.float(), refined_pred)
                else:
                    sam_pred = sam_model(image, point_coords, point_labels)
                    sam_pred_512 = F.interpolate(sam_pred, size=(512, 512), mode='bilinear', align_corners=False)
                    refined_pred_512 = refine_net(sam_pred_512)
                    refined_pred = F.interpolate(refined_pred_512, size=(image.shape[2], image.shape[3]), mode='bilinear', align_corners=False)
                    loss = 0.4 * mse_loss(refined_pred, gt2D.float()) + 0.6 * cl_dice(gt2D.float(), refined_pred)

                val_epoch_loss += loss.item()
                val_batch_metrics = SegMetrics(gt2D, refined_pred, args.metrics)
                val_iter_metrics = [val_iter_metrics[i] + val_batch_metrics[i] for i in range(len(args.metrics))]
        
        train_iter_metrics = [metric / l for metric in train_iter_metrics]
        train_metrics = {args.metrics[i]: train_iter_metrics[i] for i in range(len(train_iter_metrics))}

        val_iter_metrics = [metric / l_val for metric in val_iter_metrics]
        val_metrics = {args.metrics[i]: val_iter_metrics[i] for i in range(len(val_iter_metrics))}
        
        train_epoch_loss /= (step_train + 1)
        val_epoch_loss /= (step_val + 1)

        current_lr = optimizer.param_groups[0]['lr']
        log_epoch_results(log_path, epoch, current_lr, train_epoch_loss, val_epoch_loss, train_metrics, val_metrics, args)
        
        print_training_info(epoch, train_epoch_loss, val_epoch_loss, train_metrics, val_metrics)
        
        if scaler is not None:
            print(f"AMP Scale: {scaler.get_scale():.0f}")
        
        scheduler.step()
        
        checkpoint = {
            "model": refine_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
        }
        if scaler is not None:
            checkpoint["scaler"] = scaler.state_dict()
        
        torch.save(checkpoint, join(save_path, "refinement_model_latest.pth"))
        
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            best_val_metrics = val_metrics.copy()
            torch.save(checkpoint, join(save_path, "refinement_model_best.pth"))

    return best_val_metrics


def monitor_amp_training(scaler, loss, epoch, step):
    """Monitor AMP Training Status"""
    if scaler is not None and scaler.is_enabled():
        print(f"Epoch {epoch}, Step {step}:")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Scale: {scaler.get_scale():.0f}")
        
        # Check if gradient scaling is reduced
        if scaler.get_scale() < 1:
            print("  Warning: Gradient scaling reduced!")


if __name__ == "__main__":
    main()
