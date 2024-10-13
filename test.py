import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import  DataLoader
import torch.nn.functional as F
from segment_anything import sam_model_registry
from PIL import Image
import argparse
from tqdm import tqdm
from metrics import SegMetrics
from datasets import TestDataset
from net import SAM, RefinementNet

torch.manual_seed(114514)
torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-task_name", type=str, default="UniVesselSeg")
    parser.add_argument("-model_type", type=str, default="vit_b")
    parser.add_argument("-checkpoint_path", type=str, default="../UniVesselSeg-2024XXXX-XXXX")
    parser.add_argument("-test_data_root", type=str, default="../test")
    parser.add_argument("-batch_size", type=int, default=1)
    parser.add_argument("-num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--save_dir", type=str, default='predict', help='save predicttion')
    parser.add_argument("--metrics", nargs='+', default=['iou', 'dice'], help="metrics")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(os.path.join(args.save_dir,"initial"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir,"refine"), exist_ok=True)

    sam_checkpoint = os.path.join(args.checkpoint_path, "sam_model_best.pth")
    refine_checkpoint = os.path.join(args.checkpoint_path, "refinement_model_best.pth")
    
    sam_model = sam_model_registry[args.model_type](checkpoint=sam_checkpoint)
    sam_model = SAM(
        image_encoder=sam_model.image_encoder,
        mask_decoder=sam_model.mask_decoder,
        prompt_encoder=sam_model.prompt_encoder,
    ).to(device)

    refine_net = RefinementNet().to(device)
    with open(refine_checkpoint, "rb") as f:
        state_dict = torch.load(f, map_location=torch.device('cpu'))
        refine_net.load_state_dict(state_dict['model'])
    
    sam_model.eval()
    refine_net.eval()

    test_dataset = TestDataset(args.test_data_root)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    ori_iter_metrics = [0] * len(args.metrics)
    refine_iter_metrics = [0] * len(args.metrics)
    l = len(test_dataloader)
    with torch.no_grad():
        for image, ori_gt, original_size, img_name, point_coords, point_labels in tqdm(test_dataloader):
            image = image.to(device)
            ori_gt = ori_gt.to(device)
            point_coords = point_coords.detach().cpu().numpy()
            point_labels = point_labels.detach().cpu().numpy()

            mask = sam_model(image, point_coords, point_labels)
            mask_refine = refine_net(mask)

            # Convert to Original Image Size
            w, h = original_size
            original_size = (h, w)
            mask = F.interpolate(mask, original_size, mode="bilinear", align_corners=False)
            ori_batch_metrics = SegMetrics(ori_gt, mask, args.metrics)
            ori_iter_metrics = [ori_iter_metrics[i] + ori_batch_metrics[i] for i in range(len(args.metrics))]

            mask_refine = F.interpolate(mask_refine, original_size, mode="bilinear", align_corners=False)
            refine_batch_metrics = SegMetrics(ori_gt, mask_refine, args.metrics)
            refine_iter_metrics = [refine_iter_metrics[i] + refine_batch_metrics[i] for i in range(len(args.metrics))]

            # Save Initial Predict Mask
            mask = mask.squeeze().cpu().numpy()
            mask = (mask > 0.5).astype(np.uint8) * 255  
            mask_img = Image.fromarray(mask)          
            mask_img.save(os.path.join(args.save_dir,"initial", f"{img_name[0]}"))

            # Save Refined Mask
            mask_refine = mask_refine.squeeze().cpu().numpy()
            mask_refine = (mask_refine > 0.5).astype(np.uint8) * 255  
            mask_refine_img = Image.fromarray(mask_refine)          
            mask_refine_img.save(os.path.join(args.save_dir,"refine", f"{img_name[0]}"))
        
        ori_iter_metrics = [metric / l for metric in ori_iter_metrics]
        ori_metrics = {args.metrics[i]: '{:.4f}'.format(ori_iter_metrics[i]) for i in range(len(ori_iter_metrics))}

        refine_iter_metrics = [metric / l for metric in refine_iter_metrics]
        refine_metrics = {args.metrics[i]: '{:.4f}'.format(refine_iter_metrics[i]) for i in range(len(refine_iter_metrics))}
        print(f'IniMask metrics: {ori_metrics}, RefineMask metrics: {refine_metrics}')

if __name__ == "__main__":
    main()
