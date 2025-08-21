import os
import json
import random
import re

def create_segmentation_dataset_json(image_folder, mask_folder, train_json, val_json):
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    dataset_dict = {}
    
    for image_file in image_files:
        dataset_name = re.search(r'_(.*?)_', image_file).group(1)
        if dataset_name not in dataset_dict:
            dataset_dict[dataset_name] = []
        dataset_dict[dataset_name].append(image_file)
    
    train_dataset = {}
    val_dataset = {}

    for dataset_name, files in dataset_dict.items():

        random.shuffle(files)
        split_index = int(len(files) * 0.9)
        train_files = files[:split_index]
        val_files = files[split_index:]

        for image_file in train_files:
            image_path = os.path.join("images", image_file)
            mask_path = os.path.join("masks", image_file)
            train_dataset[image_path] = [mask_path]

        for image_file in val_files:
            image_path = os.path.join("images", image_file)
            mask_path = os.path.join("masks", image_file)
            val_dataset[image_path] = [mask_path]

    with open(train_json, 'w') as f:
        json.dump(train_dataset, f, indent=4)

    with open(val_json, 'w') as f:
        json.dump(val_dataset, f, indent=4)

image_folder = 'train/images'
mask_folder = 'train/masks'
train_json = 'train/image2label_train.json'
val_json = 'train/image2label_val.json'

create_segmentation_dataset_json(image_folder, mask_folder, train_json, val_json)