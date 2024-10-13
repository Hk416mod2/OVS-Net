import numpy as np
import random
from PIL import Image
from torch.utils.data import Dataset
import json
import os
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch
from utils.point_sampling import init_point_sampling


class VesselDataset(Dataset):
    def __init__(self, data_root, mode='train', bbox_shift=20):
        dataset = json.load(open(os.path.join(data_root, f'image2label_{mode}.json'), "r"))
        self.img_files = list(dataset.keys())
        self.gt_files = [value[0] for value in dataset.values()]
        self.point_num = 5
        
        self.bbox_shift = bbox_shift
        print(f"Number of images: {len(self.gt_files)}")

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, index):
        img_name = os.path.basename(self.gt_files[index])
        
        img_1024 = np.array(transforms.Resize((1024, 1024))(Image.open(self.img_files[index]).convert('RGB')))
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        img_1024 = img_1024 / 255.0  # Normalize to [0, 1]

        gt = np.array(transforms.Resize((1024, 1024),InterpolationMode.NEAREST)(Image.open(self.gt_files[index]).convert('L')))
        label_ids = np.unique(gt)[1:]
        gt2D = np.uint8(
            gt == random.choice(label_ids.tolist())
        )
        
        # Ensure Ground Truth is Binary
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "Ground truth should be 0, 1"
        
        # Get Prompt
        point_coords, point_labels = init_point_sampling(gt2D, self.point_num)
        
        return (
            torch.tensor(img_1024).float(),
            torch.tensor(gt2D[None, :, :]).long(),
            point_coords,
            point_labels,
            img_name,
        )
    
class TestDataset(Dataset):
    def __init__(self, data_root):
        dataset = json.load(open(os.path.join(data_root, 'image2label_test.json'), "r"))
        self.img_files = list(dataset.keys())
        self.point_num = 5
        self.gt_files = [value[0] for value in dataset.values()]
        self.bbox_shift = 20
        print(f"Number of images: {len(self.img_files)}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        img_name = os.path.basename(img_path)

        original_img = Image.open(img_path).convert('RGB')
        original_size = original_img.size

        img_1024 = np.array(transforms.Resize((1024, 1024))(original_img))
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        img_1024 = img_1024 / 255.0  # Normalize to [0, 1]

        ori_gt = Image.open(self.gt_files[index]).convert('L')

        # Get Prompt
        gt = np.array(transforms.Resize((1024, 1024),InterpolationMode.NEAREST)(ori_gt))
        label_ids = np.unique(gt)[1:]
        gt2D = np.uint8(
            gt == random.choice(label_ids.tolist())
        ) 
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "Ground truth should be 0, 1"
        point_coords, point_labels = init_point_sampling(gt2D, self.point_num)

        ori_gt = transforms.ToTensor()(ori_gt)

        return torch.tensor(img_1024).float(), ori_gt, original_size, img_name, point_coords, point_labels
