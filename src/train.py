from ultralytics import YOLO

# Load a YOLOv8 Pose model without pre-trained weights
model = YOLO("yolov8m-pose.pt")  # Use the model configuration file instead of a pre-trained weights file

# Train the model
model.train(
    data="/home/bb/Dev/6DoFPoseEstimation/yolo_v8pose_dataset_large/dataset.yaml",  # Path to your dataset.yaml
    epochs=500,          # Number of training epochs
    imgsz=640,          # Image size
    batch=32,            # Adjust based on your GPU memory
    device=[0, 1, 2, 3], # Specify GPU devices
    workers=32,            # Number of workers for data loading
    name  = 'pfl_keypoints_m',
    cache = True,
    patience = 50
        
)
