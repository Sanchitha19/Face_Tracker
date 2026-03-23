import cv2
import numpy as np
import os
import insightface
from insightface.app import FaceAnalysis

class FaceEmbedder:
    def __init__(self, model_name: str = "buffalo_l", device: str = "cuda"):
        """
        Initializes InsightFace FaceAnalysis for generating embeddings.
        """
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
        
        # Initialize InsightFace FaceAnalysis
        self.app = FaceAnalysis(name=model_name, providers=providers)
        
        # Call app.prepare(ctx_id=0, det_size=(640, 640))
        # Note: ctx_id=0 for first GPU, -1 for CPU
        ctx_id = 0 if device == "cuda" else -1
        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

    def get_embedding(self, frame: np.ndarray, bbox: list) -> np.ndarray:
        """
        Extracts face embedding from frame region defined by bbox.
        """
        if frame is None or not bbox:
            return None
            
        x1, y1, x2, y2 = map(int, bbox)
        frame_h, frame_w = frame.shape[:2]
        
        # Add 30px padding (clamped)
        pad = 30
        px1 = max(0, x1 - pad)
        py1 = max(0, y1 - pad)
        px2 = min(frame_w, x2 + pad)
        py2 = min(frame_h, y2 + pad)
        
        # Crop ROI
        crop = frame[py1:py2, px1:px2]
        
        # Min crop size Check (e.g. 20x20)
        h, w = crop.shape[:2]
        if h < 20 or w < 20:
            return None
            
        # Run InsightFace on crop
        try:
            # Need to use get_faces or simply use the app's internal methods
            # FaceAnalysis.get() expects full-sized frames but we can provide cropped regions
            faces = self.app.get(crop)
            if not faces:
                return None
                
            # Take the largest face found in crop (highest chance of correct match)
            # Embedding should be 512-d float32 normalized L2
            # InsightFace's Face object contains .normed_embedding
            best_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            return best_face.normed_embedding
            
        except Exception as e:
            # log or handle error
            print(f"Error extracting embedding: {e}")
            return None
