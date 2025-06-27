import os
import glob
import json
from PIL import Image
import shutil
from tqdm import tqdm


SOURCE_ROOT = "/home/bb/Dev/EPL-Center-Circle/set03/SPIDERCAM"
TARGET_ROOT = "/home/bb/Dev/EPL-Center-Circle/data"

#Traverse all the match folders
for match_folder in os.listdir(SOURCE_ROOT):
    match_path = os.path.join(SOURCE_ROOT, match_folder)
    if not os.path.isdir(match_path):
        continue
    
    merged_images_pth = os.path.join(TARGET_ROOT, match_folder)
    os.makedirs(merged_images_pth, exist_ok=True)

    merged_annotations = {}

    for root, dirs, files in os.walk(match_path):
        if 'annotations.json' in files:

            annot_path = os.path.join(root, 'annotations.json')
            with open(annot_path, 'r') as f:
                annotations = json.load(f)

            
            for img_name, kps in annotations.items():
                merged_annotations[img_name] = kps
                img_path = os.path.join(root, img_name)
                new_img_path = os.path.join(merged_images_pth, img_name)

                if os.path.exists(img_path):
                    shutil.copy(img_path, new_img_path)
                else:
                    print(f"Image {img_path} does not exist.")
                    continue


        with open(os.path.join(merged_images_pth, 'annotations.json'), 'w') as f:
            json.dump(merged_annotations, f, indent=4)


print("All images and annotations have been merged successfully.")
            




