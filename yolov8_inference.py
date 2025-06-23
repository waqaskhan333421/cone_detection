from ultralytics import YOLO
import cv2
import os

class YOLOv8Inference:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def infer(self, image_path):
        image = cv2.imread(image_path)
        results = self.model(image)
        return results

    def save_results(self, results, output_path):
        # If results is a list, use the first result
        if isinstance(results, list):
            result = results[0]
        else:
            result = results
        annotated_img = result.plot()
        import cv2
        cv2.imwrite(output_path, annotated_img)

    def infer_webcam(self, output_dir="webcam_results"):
        cap = cv2.VideoCapture(0)
        os.makedirs(output_dir, exist_ok=True)
        frame_id = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame)
            annotated_frame = results[0].plot()
            cv2.imshow("YOLOv8 Webcam", annotated_frame)
            # Save each frame with prediction
            cv2.imwrite(os.path.join(output_dir, f"frame_{frame_id}.jpg"), annotated_frame)
            frame_id += 1
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

def perform_inference(model_path, image_path, output_path):
    yolo = YOLOv8Inference(model_path)
    results = yolo.infer(image_path)
    yolo.save_results(results, output_path)

def main(image_path, model_path, output_path, use_webcam=False):
    yolo_inference = YOLOv8Inference(model_path)
    if use_webcam:
        yolo_inference.infer_webcam()
    else:
        perform_inference(model_path, image_path, output_path)

if __name__ == "__main__":
    model_path = r"D:\Ayaz\projects\New folder (2)\Cone Detector Computer Vision Project\runs\detect\train2\weights\best.pt"  # Path to your trained model weights
    image_path = r"D:\Ayaz\projects\New folder (2)\Cone Detector Computer Vision Project\Cone-Detector-1\test\images\amz_00016_jpg.rf.08cbc19a1d9de613a9509df89499e00c.jpg"  # Replace with your test image path
    output_path = r"D:\Ayaz\projects\New folder (2)\Cone Detector Computer Vision Project\output\result.jpg"  # Replace with your desired output path
    use_webcam = False  # Set to True to use webcam, False for image
    main(image_path, model_path, output_path, use_webcam)