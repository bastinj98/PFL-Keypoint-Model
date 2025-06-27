import os
import json
import shutil
import random
from PIL import Image
from tqdm import tqdm

# Paths
root_dir = "/home/bb/Dev/6DoFPoseEstimation/Data/Cam5_Jib_1stFight"
output_dir = "/home/bb/Dev/6DoFPoseEstimation/yolo_v8pose_corrected_dataset"
img_output_dir_train = os.path.join(output_dir, "images", "train")
img_output_dir_val = os.path.join(output_dir, "images", "val")
label_output_dir_train = os.path.join(output_dir, "labels", "train")
label_output_dir_val = os.path.join(output_dir, "labels", "val")

# Create output directories
os.makedirs(img_output_dir_train, exist_ok=True)
os.makedirs(img_output_dir_val, exist_ok=True)
os.makedirs(label_output_dir_train, exist_ok=True)
os.makedirs(label_output_dir_val, exist_ok=True)

# Configuration
train_ratio = 0.8
num_keypoints = 40

def normalize(value, max_value):
    return value / max_value

def clamp(val, min_val=0.0, max_val=1.0):
    return max(min_val, min(max_val, val))

def convert_frame_folder(frame_folder_path, frame_id):
    frame_json_path = os.path.join(frame_folder_path, "FrameData.json")
    clean_img_path = os.path.join(frame_folder_path, "CleanImage.png")

    try:
        with open(frame_json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Skipping frame {frame_id}: JSON load error - {e}")
        return None

    keypoints = data.get('KeyPoints', [])

    try:
        img = Image.open(clean_img_path)
    except Exception as e:
        print(f"Skipping frame {frame_id}: image load error - {e}")
        return None

    img_w, img_h = img.size

    if keypoints:
        x_coords = [kp['2D']['x'] for kp in keypoints]
        y_coords = [kp['2D']['y'] for kp in keypoints]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        x_center = clamp(normalize((x_min + x_max) / 2, img_w))
        y_center = clamp(normalize((y_min + y_max) / 2, img_h))
        w = clamp(normalize(x_max - x_min, img_w))
        h = clamp(normalize(y_max - y_min, img_h))
    else:
        x_center = y_center = w = h = 0.0

    keypoint_entries = []
    for kp in keypoints:
        x = normalize(kp['2D']['x'], img_w)
        y = normalize(kp['2D']['y'], img_h)

        if x > 1.0 or y > 1.0:
            print(f"Frame {frame_id}: keypoint ({x:.4f}, {y:.4f}) out of bounds, clamping")

        x = clamp(x)
        y = clamp(y)
        v = 2  # Assume all keypoints are visible
        keypoint_entries.extend([f"{x:.6f}", f"{y:.6f}", str(v)])

    while len(keypoint_entries) < num_keypoints * 3:
        keypoint_entries.extend(["0.000000", "0.000000", "0"])

    line = f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f} " + " ".join(keypoint_entries)

    if len(line.split()) != 125:
        print(f"Skipping frame {frame_id}: incorrect number of label entries ({len(line.split())})")
        return None

    label_filename = f"frame_{frame_id}.txt"
    return (line, clean_img_path, label_filename)

def split_train_val(files, train_ratio):
    if len(files) < 2:
        return files, []
    random.shuffle(files)
    split_index = max(1, int(len(files) * train_ratio))
    if split_index >= len(files):
        split_index = len(files) - 1
    return files[:split_index], files[split_index:]

def copy_files(files, img_output_dir, label_output_dir):
    for line, img_path, label_filename in files:
        label_path = os.path.join(label_output_dir, label_filename)
        with open(label_path, 'w') as f:
            f.write(line + "\n")

        img_output_path = os.path.join(img_output_dir, label_filename.replace(".txt", ".png"))
        shutil.copy(img_path, img_output_path)

def main():
    frame_folders = []

    for entry in sorted(os.listdir(root_dir)):
        entry_path = os.path.join(root_dir, entry)
        if os.path.isdir(entry_path) and entry.isdigit():
            frame_folders.append(entry_path)

    print(f"Found {len(frame_folders)} frame folders.")

    files = []
    for frame_folder_path in tqdm(frame_folders, desc="Processing frames"):
        frame_id = os.path.basename(frame_folder_path)
        result = convert_frame_folder(frame_folder_path, frame_id)
        if result:
            files.append(result)

    print(f"Total valid labeled frames: {len(files)}")

    train_files, val_files = split_train_val(files, train_ratio=train_ratio)

    print(f"Train files: {len(train_files)}")
    print(f"Val files: {len(val_files)}")

    if len(val_files) == 0:
        print("⚠️ Warning: No validation files generated. Check dataset size or train/val split ratio.")

    copy_files(train_files, img_output_dir_train, label_output_dir_train)
    copy_files(val_files, img_output_dir_val, label_output_dir_val)

    print("✅ Dataset generation complete.")

if __name__ == "__main__":
    main()
