import os
import json
import random
from sklearn.model_selection import KFold

def create_kfold_segmentation_dataset_json(image_folder, mask_folder, out_dir, n_splits=5, seed=42):
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    image_files = sorted(image_files)
    random.Random(seed).shuffle(image_files)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    
    all_folds_data = {
        "metadata": {
            "n_splits": n_splits,
            "seed": seed,
            "total_images": len(image_files),
            "image_folder": image_folder,
            "mask_folder": mask_folder
        },
        "folds": {}
    }

    for fold, (train_idx, val_idx) in enumerate(kf.split(image_files)):
        train_files = [image_files[i] for i in train_idx]
        val_files = [image_files[i] for i in val_idx]

        train_dataset = {}
        for image_file in train_files:
            # 使用相对于out_dir的路径
            image_path = os.path.join("images", image_file)
            mask_paths = [os.path.join("masks", f) for f in os.listdir(mask_folder) if f.startswith(image_file)]
            train_dataset[image_path] = mask_paths

        val_dataset = {}
        for image_file in val_files:
            # 使用相对于out_dir的路径
            image_path = os.path.join("images", image_file)
            mask_paths = [os.path.join("masks", f) for f in os.listdir(mask_folder) if f.startswith(image_file)]
            val_dataset[image_path] = mask_paths

        all_folds_data["folds"][f"fold_{fold}"] = {
            "train": train_dataset,
            "val": val_dataset,
            "train_count": len(train_files),
            "val_count": len(val_files)
        }
        
        print(f"Fold {fold}: train {len(train_files)}, val {len(val_files)}")

    output_json = os.path.join(out_dir, "train_val_fold.json")
    with open(output_json, 'w') as f:
        json.dump(all_folds_data, f, indent=2)
    
    print(f"\nAll folds data saved to: {output_json}")
    print(f"Total folds: {n_splits}")
    print(f"Total images: {len(image_files)}")
    
    return output_json

if __name__ == "__main__":
    image_folder = 'train/images'
    mask_folder = 'train/masks'
    out_dir = 'train' 
    os.makedirs(out_dir, exist_ok=True)
    create_kfold_segmentation_dataset_json(image_folder, mask_folder, out_dir, n_splits=5, seed=42)