import os
import logging
import cv2
import numpy as np
from datetime import datetime
from src import database

class EventLogger:
    def __init__(self, log_file_path: str, image_quality: int):
        """
        Initializes Python logging with both FileHandler and StreamHandler.
        """
        self.image_quality = image_quality
        self.log_file_path = log_file_path
        
        # Create directories
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        os.makedirs("logs/entries", exist_ok=True)
        os.makedirs("logs/exits", exist_ok=True)

        # Set up logger
        self.logger = logging.getLogger("FaceTracker")
        self.logger.setLevel(logging.INFO)
        
        # Avoid redundant handlers if logger exists
        if not self.logger.handlers:
            # Format [YYYY-MM-DD HH:MM:SS] [LEVEL] message
            formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', 
                                          datefmt='%Y-%m-%d %H:%M:%S')

            # File Handler
            fh = logging.FileHandler(log_file_path)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

            # Stream Handler
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            self.logger.addHandler(sh)

    def log_entry(self, face_uuid: str, track_id: int, 
                  frame: np.ndarray, bbox: list, timestamp: str,
                  conn) -> str:
        """
        Saves cropped entry image and logs event to DB and file.
        """
        date_str = timestamp.split(' ')[0]
        save_dir = os.path.join("logs", "entries", date_str)
        os.makedirs(save_dir, exist_ok=True)
        
        # Safe filename
        ts_safe = timestamp.replace(":", "_").replace(" ", "_")
        filename = f"{face_uuid}_{ts_safe}_{track_id}.jpg"
        img_path = os.path.join(save_dir, filename)
        
        # Save cropped image
        self._save_crop(frame, bbox, img_path)
        
        # DB log
        database.log_event(conn, face_uuid, "entry", timestamp, img_path, track_id)
        
        # File log
        # [ENTRY] face_uuid={face_uuid} track_id={track_id} time={timestamp} img={path}
        self.logger.info(f"[ENTRY] face_uuid={face_uuid} track_id={track_id} time={timestamp} img={img_path}")
        
        return img_path

    def log_exit(self, face_uuid: str, track_id: int, 
                 frame: np.ndarray, bbox: list, timestamp: str,
                 conn, frames_absent: int = 60) -> str:
        """
        Saves cropped exit image and logs event to DB and file.
        Matches final exit log format required by customer.
        """
        date_str = timestamp.split(' ')[0]
        last_seen_time = timestamp.split(' ')[1]
        save_dir = os.path.join("logs", "exits", date_str)
        os.makedirs(save_dir, exist_ok=True)
        
        ts_safe = timestamp.replace(":", "_").replace(" ", "_")
        filename = f"{face_uuid}_{ts_safe}_{track_id}.jpg"
        img_path = os.path.join(save_dir, filename)
        
        # Save cropped image (using last good frame)
        self._save_crop(frame, bbox, img_path)
        
        # DB log
        database.log_event(conn, face_uuid, "exit", timestamp, img_path, track_id)
        
        # File log
        # [EXIT] face_uuid=abc12345 track_id=3 last_seen=14:32:10 frames_absent=60 img=...
        self.logger.info(f"[EXIT] face_uuid={face_uuid} track_id={track_id} last_seen={last_seen_time} frames_absent={frames_absent} img={img_path}")
        
        return img_path

    def log_system(self, message: str) -> None:
        """
        Writes INFO-level message to system logs.
        """
        self.logger.info(message)

    def _save_crop(self, frame, bbox, path):
        """
        Helper method to crop face from frame with padding and save to path.
        """
        if frame is None or not bbox:
            return
            
        x1, y1, x2, y2 = map(int, bbox)
        h, w = frame.shape[:2]
        
        # Add 10px padding
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return

        cv2.imwrite(path, crop, [int(cv2.IMWRITE_JPEG_QUALITY), self.image_quality])
