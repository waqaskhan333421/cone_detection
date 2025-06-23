from ultralytics import YOLO

if __name__ == "__main__":
    # Load a larger YOLOv8 model for better accuracy (optional)
    model = YOLO('yolov8s.pt')

    # Train the model using GPU (device=0)
    model.train(
        data='dataset/data.yaml',  # Path to your data.yaml
        epochs=100,                # More epochs for better results
        imgsz=640,                 # Image size
        batch=16,                  # Batch size
        device=0                   # 0 for first GPU, 'cpu' for CPU
    )