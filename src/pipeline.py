import cv2
import numpy as np
import time
import os
from datetime import datetime
from src.config_loader import get_config
from src.database import init_db, get_unique_visitor_count, upsert_visitor_summary, update_last_seen
from src.detector import FaceDetector
from src.embedder import FaceEmbedder
from src.recognizer import FaceRecognizer
from src.tracker import FaceTracker
from src.logger import EventLogger


class Pipeline:
    def __init__(self, config: dict):
        """
        Main pipeline that coordinates all components.
        """
        self.config = config
        
        # Init DB connection
        self.conn = init_db(config["database"]["path"])

        # Create sub-components
        device = "cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu"
        
        # Detector
        self.detector = FaceDetector(
            model_path=config["detection"]["yolo_model"],
            conf=config["detection"]["confidence_threshold"],
            iou=config["detection"]["iou_threshold"],
            device=device
        )
        
        # Embedder
        self.embedder = FaceEmbedder(
            model_name=config["recognition"]["embedding_model"],
            device=device
        )
        
        # Recognizer
        self.recognizer = FaceRecognizer(
            conn=self.conn,
            similarity_threshold=config["recognition"]["similarity_threshold"]
        )
        
        # Tracker
        self.tracker = FaceTracker(
            max_disappeared=config["tracking"]["max_disappeared"],
            iou_threshold=config["tracking"]["iou_threshold"],
            exit_confirm_frames=config["tracking"].get("exit_confirm_frames", 180)
        )
        
        # Logger
        self.logger = EventLogger(
            log_file_path=config["logging"]["log_file"],
            image_quality=config["logging"]["image_quality"]
        )
        
        self.session_start = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.frame_count = 0
        self.total_events = 0

    def run(self) -> None:
        """
        Main run loop for video capture and processing.
        Includes reconnection logic for RTSP streams.
        """
        use_rtsp = self.config.get("use_rtsp", False)
        video_src = self.config["rtsp_url"] if use_rtsp else self.config["video_source"]
        
        cap = cv2.VideoCapture(video_src)
        reconnect_attempts = 0
        max_reconnect = 5
        
        if not cap.isOpened():
            print(f"ERROR: Failed to open source {video_src}")
            self.logger.log_system(f"Failed to open source {video_src}")
            return

        self.logger.log_system(f"Pipeline started on {'RTSP' if use_rtsp else 'File'}: {video_src}")
        print(f"Pipeline started on {'RTSP' if use_rtsp else 'File'}: {video_src}")

        while True:
            ret, frame = cap.read()
            
            if not ret:
                if use_rtsp and reconnect_attempts < max_reconnect:
                    reconnect_attempts += 1
                    print(f"RTSP connection lost. Reconnecting ({reconnect_attempts}/{max_reconnect})...")
                    self.logger.log_system(f"RTSP reconnection attempt {reconnect_attempts}")
                    
                    cap.release()
                    time.sleep(2)
                    cap = cv2.VideoCapture(video_src)
                    continue
                else:
                    # End of video file or exhausted RTSP retries
                    break
            
            # Reset reconnect attempts on success
            reconnect_attempts = 0
            
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.frame_count += 1
            
            # Skip frames logic
            skip_frames = self.config.get("skip_frames", 0)
            if self.frame_count % (skip_frames + 1) == 0:
                processed_frame = self._process_frame(frame, self.frame_count)
            else:
                # Update tracker with empty detections to maintain track counts
                # This ensures exit timers keep running even on skipped frames
                active_track_returns = self.tracker.update([], frame, timestamp)
                # Still check for exits even on skipped frames
                for track in active_track_returns:
                    if track.get("just_exited"):
                        self._handle_exit(track, frame, timestamp)
                processed_frame = self._draw_overlays(frame, active_track_returns)
            
            # Show output
            if self.config["display"]["show_video"]:
                cv2.imshow("Intelligent Face Tracker", processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        self.cleanup()
        cap.release()

    def _process_frame(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """
        Processes frame and handles state-based entry/exit logging.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        detections = self.detector.detect(frame)
        active_track_returns = self.tracker.update(detections, frame, timestamp)
        
        for track in active_track_returns:
            track_id = track["track_id"]
            
            # Step 1/2/3: Identity Resolution for NEW tracks
            if track.get("is_new"):
                bbox = track["bbox"]
                embedding = self.embedder.get_embedding(frame, bbox)
                if embedding is None:
                    continue
                
                # Run 4-step check (Check memory tracks + Check DB)
                face_uuid, score = self.recognizer.identify(embedding, self.tracker)
                
                if face_uuid is None:
                    # New visitor (Step 4)
                    face_uuid = self.recognizer.register_new_face(embedding, timestamp)
                else:
                    # Returning visitor (Step 2 or 3)
                    # Note: update_last_seen increments visit_count
                    from src.database import update_last_seen
                    update_last_seen(self.conn, face_uuid, timestamp)
                
                # Store identity and embedding on internal tracker object
                self.tracker.tracks[track_id]["face_uuid"] = face_uuid
                self.tracker.tracks[track_id]["embedding"] = embedding
                
                # Log ENTRY
                self._handle_entry(self.tracker.tracks[track_id], frame, timestamp)

            # Handle Re-entry (Pending Exit -> Active)
            elif track.get("re_entry"):
                face_uuid = track["face_uuid"]
                if face_uuid:
                    from src.database import update_last_seen
                    update_last_seen(self.conn, face_uuid, timestamp)
                    self._handle_entry(self.tracker.tracks[track_id], frame, timestamp)

            # Handle Exit (Pending Exit -> Exited)
            elif track.get("just_exited"):
                self._handle_exit(track, frame, timestamp)

        annotated = self._draw_overlays(frame, active_track_returns)
        
        # Periodic debug save
        if frame_number % 100 == 0:
            os.makedirs("logs/annotated_frames", exist_ok=True)
            cv2.imwrite(f"logs/annotated_frames/frame_{frame_number}.jpg", annotated)
            
        return annotated

    def _handle_entry(self, track_data: dict, frame: np.ndarray, timestamp: str) -> None:
        """
        Logs entry event with visit count.
        """
        face_uuid = track_data["face_uuid"]
        track_id = track_data["track_id"]
        from src.database import get_visit_count
        visit_num = get_visit_count(self.conn, face_uuid)
        
        img_path = self.logger.log_entry(face_uuid, track_id, frame, track_data["bbox"], timestamp, self.conn)
        self.total_events += 1
        track_data["entry_logged"] = True
        
        print(f"[ENTRY] face_uuid={face_uuid[:8]} visit={visit_num} track_id={track_id} time={timestamp.split(' ')[1]}")

    def _handle_exit(self, track_data: dict, frame: np.ndarray, timestamp: str) -> None:
        """
        Logs exit event using last good frame.
        """
        face_uuid = track_data.get("face_uuid")
        track_id = track_data.get("track_id")
        if not face_uuid:
            return
            
        last_good_frame = track_data.get("last_good_frame")
        last_good_bbox = track_data.get("last_good_bbox")
        last_ts = track_data.get("last_good_timestamp")
        
        from src.database import get_visit_count
        visit_num = get_visit_count(self.conn, face_uuid)

        # Confirm crop from last seen
        img_path = self.logger.log_exit(
            face_uuid, track_id, last_good_frame, last_good_bbox, last_ts, 
            self.conn, frames_absent=track_data.get("disappeared", 0)
        )
        self.total_events += 1
        
        last_ts_str = last_ts.split(' ')[1] if last_ts else "??:??:??"
        print(f"[EXIT]  face_uuid={face_uuid[:8]} visit={visit_num} track_id={track_id} time={last_ts_str}")

    def _draw_overlays(self, frame: np.ndarray, active_returns: list) -> np.ndarray:
        """
        Draws boxes/labels for currently ACTIVE tracks only.
        """
        draw_frame = frame.copy()
        for track in active_returns:
            if track.get("just_exited") or track.get("state") == "pending_exit":
                continue
                
            x1, y1, x2, y2 = map(int, track["bbox"])
            face_uuid = track.get("face_uuid") or "Unknown"
            tid = track["track_id"]
            
            f_short = face_uuid[:8] if face_uuid != "Unknown" else "Unknown"
            label = f"ID: {f_short} | T:{tid}"
            cv2.rectangle(draw_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(draw_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        unique_count = get_unique_visitor_count(self.conn)
        cv2.putText(draw_frame, f"Unique Visitors: {unique_count}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return draw_frame
    def get_visitor_count(self) -> int:
        """
        Maintained for main.py compatibility.
        """
        from src.database import get_unique_visitor_count
        return get_unique_visitor_count(self.conn)

    def cleanup(self) -> None:
        """
        Force exit for all remaining tracks and print final summary.
        """
        session_end = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Force EXIT for all active or pending tracks
        for tid, tdata in list(self.tracker.tracks.items()):
            if tdata.get("face_uuid"):
                self._handle_exit(tdata, None, session_end)

        # Final Summary Print
        from src.database import get_unique_visitor_count, get_all_faces_summary
        unique_v = get_unique_visitor_count(self.conn)
        entries = 0
        exits = 0
        # Get counts from events table
        cursor = self.conn.cursor()
        cursor.execute("SELECT event_type, COUNT(*) FROM events GROUP BY event_type")
        for row in cursor.fetchall():
            if row[0] == 'entry': entries = row[1]
            if row[0] == 'exit': exits = row[1]

        print("\n" + "═"*40)
        print("SESSION SUMMARY")
        print("═"*40)
        print(f"Total unique visitors  : {unique_v}")
        print(f"Total entry events     : {entries}")
        print(f"Total exit events      : {exits}")
        print("─"*40)
        print("Per-person breakdown:")
        print(f"{'face_uuid':<12} | {'visits':<6} | {'first_seen':<8} | {'last_seen':<8}")
        
        summary_rows = get_all_faces_summary(self.conn)
        for f_uuid, visits, first, last in summary_rows:
            f_short = f_uuid[:8]
            first_t = first.split(' ')[1]
            last_t = last.split(' ')[1]
            print(f"{f_short:<12} | {visits:<6} | {first_t:<10} | {last_t:<10}")
        print("═"*40 + "\n")
        
        cv2.destroyAllWindows()
