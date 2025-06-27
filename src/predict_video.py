import cv2
from ultralytics import YOLO
import numpy as np
import os
import argparse

def load_model(model_path: str):
    print(f"Loading model from {model_path}")
    return YOLO(model_path)

def draw_keypoints(image, keypoints, connections=None, color=(0, 255, 0)):
    for kp in keypoints:
        if kp[2] > 0.2:  # Confidence threshold
            cv2.circle(image, (int(kp[0]), int(kp[1])), 3, color, -1)
    
    if connections:
        for i, j in connections:
            if keypoints[i][2] > 0.2 and keypoints[j][2] > 0.2:
                pt1 = tuple(map(int, keypoints[i][:2]))
                pt2 = tuple(map(int, keypoints[j][:2]))
                cv2.line(image, pt1, pt2, color, 2)

def process_frame(frame, model):
    results = model.predict(source=frame, verbose=False, conf=0.8)
    
    
    annotated_frame = frame.copy()

    for result in results:
        if hasattr(result, "keypoints") and result.keypoints is not None:
            for kp_set in result.keypoints.xy:
                kp_array = kp_set.cpu().numpy()
                keypoints = np.hstack([kp_array, np.ones((kp_array.shape[0], 1))])  # [x, y, conf]
                draw_keypoints(annotated_frame, keypoints)
    
    return annotated_frame

def process_video(input_path: str, output_path: str, model_path: str):
    model = load_model(model_path)
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {input_path}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    print(f"Saving output to {output_path}")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        annotated = process_frame(frame, model)
        out.write(annotated)

        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"Processed frame {frame_idx}")

    cap.release()
    out.release()
    print("Done.")

def main():
    parser = argparse.ArgumentParser(description="Run video processing using a PyTorch model.")
    parser.add_argument("--input_video", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output video")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the PyTorch .pt model")

    args = parser.parse_args()

    process_video(args.input_video, args.output_path, args.model_path)

if __name__ == "__main__":
    main()



'''

python predict_video.py \
  --input_video /home/bb/Dev/PFL-Keypoint-Model/data/sample_input_video.mp4 \
  --output_path /home/bb/Dev/PFL-Keypoint-Model/output \
  --model_path /home/bb/Dev/PFL-Keypoint-Model/models/KeyPoint_YOLOv8m-Pose_500eps.pt


'''