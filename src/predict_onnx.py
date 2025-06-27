import cv2
import numpy as np
import onnxruntime as ort
import argparse
import os


def preprocess(frame, input_size):
    # Resize frame to model input size and normalize to [0,1]
    frame_resized = cv2.resize(frame, input_size)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    input_tensor = frame_rgb.astype(np.float32) / 255.0
    # Change HWC to CHW format
    input_tensor = np.transpose(input_tensor, (2, 0, 1))
    # Add batch dimension
    input_tensor = np.expand_dims(input_tensor, axis=0)
    return input_tensor

def draw_keypoints(frame, keypoints, conf_threshold=0.2):
    for kp in keypoints:
        x, y = kp[0], kp[1]
        # Assuming no confidence or confidence is embedded elsewhere, just draw
        cv2.circle(frame, (int(x), int(y)), 3, (0, 255, 0), -1)

def run_onnx_inference(onnx_path, video_path, output_path, input_size=(1280,1280)):
    # Load ONNX model
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    print(input_name)
    output_name = session.get_outputs()[0].name
    print(output_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_path, f"{video_name}_keypointOutput.mp4")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess frame for model input
        input_tensor = preprocess(frame, input_size)

        # Run inference
        outputs = session.run([output_name], {input_name: input_tensor})
        print(outputs)
        # The output shape/format depends on your model
        # Let's assume output[0] shape: (batch, num_keypoints*2)
        preds = outputs[0]  # shape e.g. (1, num_keypoints*2)
        preds = preds[0]    # remove batch dim

        num_keypoints = preds.shape[0] // 2
        keypoints = []
        for i in range(num_keypoints):
            x_norm = preds[2*i].item()
            y_norm = preds[2*i + 1].item()
            x_px = x_norm * width
            y_px = y_norm * height
            keypoints.append((x_px, y_px))


        # Draw keypoints on original frame
        draw_keypoints(frame, keypoints)

        # Write frame and show
        out.write(frame)
        cv2.imshow("Keypoints ONNX", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Done. Output saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Run 6DoF pose estimation on a video using ONNX model.")
    parser.add_argument("--input_video", type=str, required=True, help="Path to the input video file")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output video")
    parser.add_argument("--onnx_model_path", type=str, required=True, help="Path to the ONNX model file")

    args = parser.parse_args()

    run_onnx_inference(args.onnx_model_path, args.input_video, args.output_path)

if __name__ == "__main__":
    main()


'''
python predict_onnx.py \
  --input_video /home/bb/Dev/PFL-Keypoint-Model/data/sample_input_video.mp4 \
  --output_path /home/bb/Dev/PFL-Keypoint-Model/output \
  --onnx_model_path /home/bb/Dev/PFL-Keypoint-Model/models/KeyPoint_PFL_YOLOv8m_Pose_small.onnx

'''
