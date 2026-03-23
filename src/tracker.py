import numpy as np

class FaceTracker:
    def __init__(self, max_disappeared: int, iou_threshold: float, exit_confirm_frames: int = 180):
        """
        Maintains internal dict: self.tracks = {track_id: track_data}
        Each track follows a state machine: active -> pending_exit -> exited.
        """
        self.max_disappeared = max_disappeared
        self.exit_confirm_frames = exit_confirm_frames
        self.iou_threshold = iou_threshold
        self.next_track_id = 1
        self.tracks = {}

    def update(self, detections: list, frame: np.ndarray, timestamp: str) -> list:
        """
        Match current detections to existing tracks using IoU.
        Update states according to transitions.
        """
        active_track_returns = []
        matched_detections = set()
        matched_tracks = set()

        # 1. Match current detections to existing tracks (active or pending_exit)
        # We process tracks in reverse order of disappearance (active first)
        sorted_track_ids = sorted(self.tracks.keys(), key=lambda x: self.tracks[x]["disappeared"])
        
        for track_id in sorted_track_ids:
            track_data = self.tracks[track_id]
            if track_data["state"] == "exited":
                continue
                
            best_iou = -1.0
            best_det_idx = -1
            
            for i, det in enumerate(detections):
                if i in matched_detections:
                    continue
                
                iou = self.get_iou(track_data["bbox"], det["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = i
            
            if best_iou >= self.iou_threshold:
                # Transition: any -> active
                re_entry = (track_data["state"] == "pending_exit")
                
                track_data["state"] = "active"
                track_data["bbox"] = detections[best_det_idx]["bbox"]
                track_data["disappeared"] = 0
                track_data["last_good_frame"] = frame.copy()
                track_data["last_good_bbox"] = track_data["bbox"]
                track_data["last_good_timestamp"] = timestamp
                
                matched_detections.add(best_det_idx)
                matched_tracks.add(track_id)
                
                active_track_returns.append({
                    **track_data,
                    "re_entry": re_entry,
                    "is_new": False
                })

        # 2. Handle unmatched tracks (Increment disappeared + transitions)
        to_delete = []
        for track_id, track_data in self.tracks.items():
            if track_id not in matched_tracks:
                track_data["disappeared"] += 1
                
                # State: active -> pending_exit
                if track_data["state"] == "active" and track_data["disappeared"] > self.max_disappeared:
                    track_data["state"] = "pending_exit"
                    # Mark entry_logged=False so when they return it triggers entry
                    track_data["entry_logged"] = False
                
                # State: pending_exit -> exited
                if track_data["state"] == "pending_exit" and track_data["disappeared"] > self.exit_confirm_frames:
                    track_data["state"] = "exited"
                    # We will return this track with exited signal then delete it
                    active_track_returns.append({
                        **track_data,
                        "just_exited": True,
                        "is_new": False
                    })
                    to_delete.append(track_id)

        # 3. Handle new tracks
        for i, det in enumerate(detections):
            if i not in matched_detections:
                track_id = self.next_track_id
                self.next_track_id += 1
                
                new_track = {
                    "track_id": track_id,
                    "face_uuid": None,
                    "bbox": det["bbox"],
                    "disappeared": 0,
                    "state": "active",
                    "last_good_frame": frame.copy(),
                    "last_good_bbox": det["bbox"],
                    "last_good_timestamp": timestamp,
                    "entry_logged": False
                }
                
                self.tracks[track_id] = new_track
                
                active_track_returns.append({
                    **new_track,
                    "is_new": True,
                    "re_entry": False
                })

        # 4. Remove exited tracks from memory
        for tid in to_delete:
            del self.tracks[tid]
            
        return active_track_returns

    @staticmethod
    def get_iou(boxA, boxB) -> float:
        """
        Calculate Intersection over Union (IoU) of two bounding boxes.
        box: [x1, y1, x2, y2]
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
