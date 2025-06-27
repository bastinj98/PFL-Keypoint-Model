import os
import cv2
import albumentations as A
from tqdm import tqdm
import shutil

# Define the augmentation pipeline
transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.HueSaturationValue(p=0.5),
    A.RGBShift(p=0.5),
    A.ChannelShuffle(p=0.5),
    A.GaussNoise(p=0.5),
    A.Blur(p=0.3),
    A.MotionBlur(p=0.3),
    A.MedianBlur(p=0.3),
    A.ToGray(p=0.2),
])

# Paths
original_dataset = 'dataset'
augmented_dataset = 'dataset_aug'
subsets = ['train', 'val']

for subset in subsets:
    orig_images_dir = os.path.join(original_dataset, 'images', subset)
    orig_labels_dir = os.path.join(original_dataset, 'labels', subset)
    aug_images_dir = os.path.join(augmented_dataset, 'images', subset)
    aug_labels_dir = os.path.join(augmented_dataset, 'labels', subset)

    os.makedirs(aug_images_dir, exist_ok=True)
    os.makedirs(aug_labels_dir, exist_ok=True)

    image_files = [f for f in os.listdir(orig_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for filename in tqdm(image_files, desc=f'Processing {subset} set'):
        image_path = os.path.join(orig_images_dir, filename)
        label_path = os.path.join(orig_labels_dir, os.path.splitext(filename)[0] + '.txt')

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            continue

        # Save original image and label
        shutil.copy(image_path, os.path.join(aug_images_dir, filename))
        if os.path.exists(label_path):
            shutil.copy(label_path, os.path.join(aug_labels_dir, os.path.splitext(filename)[0] + '.txt'))

        # Apply augmentation
        augmented = transform(image=image)
        augmented_image = augmented['image']

        # Create unique filename for augmented image
        name, ext = os.path.splitext(filename)
        aug_filename = f"{name}_aug{ext}"
        aug_image_path = os.path.join(aug_images_dir, aug_filename)
        aug_label_path = os.path.join(aug_labels_dir, f"{name}_aug.txt")

        # Save augmented image
        cv2.imwrite(aug_image_path, augmented_image)

        # Copy and rename label file
        if os.path.exists(label_path):
            shutil.copy(label_path, aug_label_path)


# Count the number of images in the augmented dataset
image_extensions = ('.jpg', '.jpeg', '.png')

total_images = 0

for subset in subsets:
    images_dir = os.path.join(augmented_dataset, 'images', subset)
    if not os.path.exists(images_dir):
        print(f"Directory {images_dir} does not exist.")
        continue

    # List all files in the directory
    files = os.listdir(images_dir)

    # Filter files with image extensions
    image_files = [f for f in files if f.lower().endswith(image_extensions)]

    num_images = len(image_files)
    total_images += num_images

    print(f"{subset.capitalize()} set: {num_images} images")

print(f"Total images in dataset_aug: {total_images}")
