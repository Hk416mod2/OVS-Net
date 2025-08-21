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
        self.data_root = data_root
        self.mode = mode
        self.bbox_shift = bbox_shift
        self.point_num = 5
        
        if mode == 'train':
            json_file = os.path.join(data_root, 'image2label_train.json')
        elif mode == 'val':
            json_file = os.path.join(data_root, 'image2label_val.json')
        else:
            raise ValueError(f"Mode must be 'train' or 'val', got {mode}")
        
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"JSON file not found: {json_file}")
        
        with open(json_file, 'r') as f:
            dataset = json.load(f)
        
        self.img_files = [os.path.join(data_root, img_path) for img_path in dataset.keys()]
        self.gt_files = [os.path.join(data_root, value[0]) for value in dataset.values()]
        
        print(f"Number of {mode} images: {len(self.gt_files)}")
        print(f"JSON file: {json_file}")
        
        self._validate_files()

    def _validate_files(self):
        missing_files = []
        
        for img_file, gt_file in zip(self.img_files, self.gt_files):
            if not os.path.exists(img_file):
                missing_files.append(f"Image: {img_file}")
            if not os.path.exists(gt_file):
                missing_files.append(f"GT: {gt_file}")
        
        if missing_files:
            print(f"Warning: Found {len(missing_files)} missing files:")
            for file in missing_files[:5]:  # 只显示前5个
                print(f"  {file}")
            if len(missing_files) > 5:
                print(f"  ... and {len(missing_files) - 5} more")
        else:
            print("All files validated successfully!")

    def __len__(self):
        return len(self.gt_files)

    def __getitem__(self, index):
        img_name = os.path.basename(self.gt_files[index])
        
        img_1024 = np.array(transforms.Resize((1024, 1024))(Image.open(self.img_files[index]).convert('RGB')))
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        img_1024 = img_1024 / 255.0

        gt = np.array(transforms.Resize((1024, 1024), InterpolationMode.NEAREST)(Image.open(self.gt_files[index]).convert('L')))
        label_ids = np.unique(gt)[1:]
        
        if len(label_ids) == 0:
            gt2D = np.zeros((1024, 1024), dtype=np.uint8)
        else:
            gt2D = np.uint8(gt == random.choice(label_ids.tolist()))
        
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "Ground truth should be 0, 1"
        
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
        self.data_root = data_root
        self.point_num = 1
        
        json_file = os.path.join(data_root, 'test.json')
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Test JSON file not found: {json_file}")
        
        with open(json_file, 'r') as f:
            dataset = json.load(f)
        
        self.img_files = [os.path.join(data_root, img_path) for img_path in dataset.keys()]
        self.gt_files = [os.path.join(data_root, value[0]) for value in dataset.values()]
        
        print(f"Number of test images: {len(self.img_files)}")
        print(f"Test JSON file: {json_file}")

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):
        img_path = self.img_files[index]
        img_name = os.path.basename(img_path)

        original_img = Image.open(img_path).convert('RGB')
        original_size = original_img.size

        img_1024 = np.array(transforms.Resize((1024, 1024))(original_img))
        img_1024 = np.transpose(img_1024, (2, 0, 1))
        img_1024 = img_1024 / 255.0

        ori_gt = Image.open(self.gt_files[index]).convert('L')

        gt = np.array(transforms.Resize((1024, 1024), InterpolationMode.NEAREST)(ori_gt))
        label_ids = np.unique(gt)[1:]
        
        if len(label_ids) == 0:
            gt2D = np.zeros((1024, 1024), dtype=np.uint8)
        else:
            gt2D = np.uint8(gt == random.choice(label_ids.tolist()))
        
        assert np.max(gt2D) == 1 and np.min(gt2D) == 0.0, "Ground truth should be 0, 1"
        point_coords, point_labels = init_point_sampling(gt2D, self.point_num)

        ori_gt = transforms.ToTensor()(ori_gt)

        return torch.tensor(img_1024).float(), ori_gt, original_size, img_name, point_coords, point_labels