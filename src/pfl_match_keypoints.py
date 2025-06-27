import os
import cv2
import faiss
import pickle
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
from ultralytics import YOLO
import argparse

# === CONFIG ===
NUM_KEYPOINTS = 40
INDEX_DIM = NUM_KEYPOINTS * 2
YOLO_MODEL_PATH = "/home/bb/Dev/6DoFPoseEstimation/runs/pose/train/weights/last.pt"
FONT = ImageFont.load_default()

# === Load YOLO Model ===
model = YOLO(YOLO_MODEL_PATH)

# === Keypoint Feature Extraction ===
def extract_keypoints(img_path_or_pil):
    if isinstance(img_path_or_pil, str):
        img = cv2.imread(img_path_or_pil)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = np.array(img_path_or_pil)

    results = model.predict(img, conf=0.5)
    if len(results[0].keypoints.data) == 0:
        return None
    keypoints = results[0].keypoints.data[0][:, :2].cpu().numpy()
    if keypoints.shape[0] != NUM_KEYPOINTS:
        return None
    return keypoints

def keypoints_to_vector(kpts):
    vec = kpts.flatten()
    vec /= np.linalg.norm(vec) + 1e-8
    return vec.astype("float32")

# === Calibration Indexing ===
def build_faiss_index(calib_folder, index_path, metadata_path):
    index = faiss.IndexFlatIP(INDEX_DIM)
    metadata = []

    print("[INFO] Indexing calibration frames...")
    for fname in tqdm(os.listdir(calib_folder)):
        if not fname.lower().endswith(('.jpg', '.png')):
            continue
        path = os.path.join(calib_folder, fname)
        kpts = extract_keypoints(path)
        if kpts is None:
            continue
        vec = keypoints_to_vector(kpts)
        index.add(vec.reshape(1, -1))
        metadata.append({"filename": fname})

    faiss.write_index(index, index_path)
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"[INFO] Indexed {len(metadata)} calibration images.")

# === Sample Video Frames ===
def extract_video_frames(video_path, num_frames=20):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = [int(i * frame_count / num_frames) for i in range(num_frames)]
    frames = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
    cap.release()
    return frames

# === Matching with FAISS ===
def match_frames_to_calib(frames, index_path, metadata_path, k=5):
    index = faiss.read_index(index_path)
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)

    matches = []
    for frame in tqdm(frames, desc="Matching frames"):
        kpts = extract_keypoints(frame)
        if kpts is None:
            matches.append([])
            continue
        vec = keypoints_to_vector(kpts)
        D, I = index.search(vec.reshape(1, -1), k)
        frame_matches = [{"filename": metadata[i]["filename"], "score": float(D[0][j])} for j, i in enumerate(I[0])]
        matches.append(frame_matches)
    return matches

# === Contact Sheet ===
def create_contact_sheet(query_frames, top_matches, calib_folder, save_path="contact_sheet.jpg"):
    thumb_size = (224, 224)
    sheet_width = (1 + 5) * thumb_size[0]
    sheet_height = len(query_frames) * thumb_size[1]
    sheet = Image.new("RGB", (sheet_width, sheet_height), (255, 255, 255))

    for row_idx, (query_img, matches) in enumerate(zip(query_frames, top_matches)):
        y_offset = row_idx * thumb_size[1]
        sheet.paste(query_img.resize(thumb_size), (0, y_offset))

        for col_idx, match in enumerate(matches):
            match_path = os.path.join(calib_folder, match["filename"])
            match_img = Image.open(match_path).convert("RGB").resize(thumb_size)
            draw = ImageDraw.Draw(match_img)
            draw.rectangle([0, 0, thumb_size[0], 20], fill=(0, 0, 0))
            draw.text((5, 2), f"{match['score']:.3f}", fill="white", font=FONT)
            sheet.paste(match_img, ((col_idx + 1) * thumb_size[0], y_offset))

    sheet.save(save_path)
    print(f"[INFO] Contact sheet saved to {save_path}")



def main():
    parser = argparse.ArgumentParser(description="Match keypoint frames from video to calibration set using FAISS.")

    parser.add_argument("--calib_folder", type=str, required=True,
                        help="Path to the folder containing calibration frames.")
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to the input video file.")
    parser.add_argument("--index_path", type=str, default="faiss_index/calib_index.faiss",
                        help="Path to save or load the FAISS index.")
    parser.add_argument("--metadata_path", type=str, default="faiss_index/calib_meta.pkl",
                        help="Path to save or load the metadata file.")
    parser.add_argument("--output_sheet", type=str, default="keypoint_contact_sheet.jpg",
                        help="Path to save the contact sheet image.")
    parser.add_argument("--num_frames", type=int, default=20,
                        help="Number of frames to extract from the video.")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Number of top matches to retrieve for each frame.")

    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.index_path), exist_ok=True)

    if not os.path.exists(args.index_path) or not os.path.exists(args.metadata_path):
        build_faiss_index(args.calib_folder, args.index_path, args.metadata_path)

    frames = extract_video_frames(args.video_path, num_frames=args.num_frames)
    top_matches = match_frames_to_calib(frames, args.index_path, args.metadata_path, k=args.top_k)
    create_contact_sheet(frames, top_matches, args.calib_folder, save_path=args.output_sheet)

if __name__ == "__main__":
    main()




'''

python src/pfl_match_keypoints.py \
  --calib_folder /home/bb/Dev/PFL-Keypoint-Model/data/calib_frames \
  --video_path /home/bb/Dev/PFL-Keypoint-Model/data/sample_input_video.mp4 \
  --index_path faiss_index/calib_index.faiss \
  --metadata_path faiss_index/calib_meta.pkl \
  --output_sheet output/keypoint_contact_sheet.jpg \
  --num_frames 20 \
  --top_k 5

'''