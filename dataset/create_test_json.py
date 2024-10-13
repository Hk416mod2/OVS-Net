import os
import json
import random

def create_segmentation_dataset_json(image_folder, mask_folder, test_json):
    test_files = os.listdir(image_folder)

    test_dataset = {}

    for image_file in test_files:
        if image_file.endswith('.png'):
            image_path = os.path.join(image_folder, image_file)
            mask_paths = []
            for mask_file in os.listdir(mask_folder):
                if mask_file.startswith(image_file):
                    mask_path = os.path.join(mask_folder, mask_file)
                    mask_paths.append(mask_path)
            test_dataset[image_path] = mask_paths

    with open(test_json, 'w') as f:
        json.dump(test_dataset, f, indent=4)

image_folder = '../test/images'
mask_folder = '../test/masks'
test_json = 'image2label_test.json'

create_segmentation_dataset_json(image_folder, mask_folder, test_json)
