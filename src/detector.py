import os
import cv2
import numpy as np
from ultralytics import YOLO
from src.config_loader import get_config

class FaceDetector:
    def __init__(self, model_path: str, conf: float, iou: float, device: str = "cuda"):
        """
        Initializes the YOLOv8-face detector.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"YOLO model not found at {model_path}. "
                "Please download yolov8n-face.pt and place it in the /models/ folder."
            )
        
        self.conf = conf
        self.iou = iou
        self.device = device
        
        # Load model using ultralytics.YOLO
        try:
            self.model = YOLO(model_path)
            # Try a dummy inference to check if cuda works
            if self.device == "cuda":
                self.model.to(self.device)
        except Exception as e:
            print(f"CUDA initialization failed or unavailable: {e}")
            self.device = "cpu"
            self.model = YOLO(model_path)
            print("Switched to CPU for inference.")

        cfg = get_config()
        self.min_face_size = cfg["recognition"].get("min_face_size", 40)

    def detect(self, frame: np.ndarray) -> list:
        """
        Runs YOLO model to detect faces.
        Returns a list of dicts, one per detected face.
        """
        if frame is None:
            return []

        frame_h, frame_w = frame.shape[:2]
        
        try:
            results = self.model.predict(
                source=frame,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=False
            )
        except Exception as e:
            print(f"Error during YOLO inference: {e}")
            return []

        detections = []
        if not results:
            return []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get coords
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0].cpu().numpy())
                
                # Clamp coordinates to frame boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame_w, x2)
                y2 = min(frame_h, y2)
                
                # Check min face size
                if (x2 - x1) < self.min_face_size or (y2 - y1) < self.min_face_size:
                    continue
                
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "frame_h": frame_h,
                    "frame_w": frame_w
                })

        return detections
