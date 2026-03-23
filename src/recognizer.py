import uuid
import numpy as np
from scipy.spatial.distance import cosine
from src import database

class FaceRecognizer:
    def __init__(self, conn, similarity_threshold: float):
        """
        Initializes FaceRecognizer.
        On init, call reload_embeddings() to load all known faces from DB into memory.
        """
        self.conn = conn
        self.similarity_threshold = similarity_threshold
        self.known_faces = [] # list of {face_uuid, embedding}
        self.reload_embeddings()

    def identify(self, embedding: np.ndarray, tracker_ref=None) -> tuple:
        """
        Compute cosine similarity between input embedding and:
        1. Current tracks in tracker (Active & Pending Exit)
        2. Database of known faces
        Return (face_uuid, similarity_score)
        """
        if embedding is None:
            return (None, 0.0)

        # L2 Normalization
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        # Step 2: Check tracks in memory (Active or Pending Exit)
        best_match = None
        best_score = -1.0
        
        if tracker_ref:
            for track_id, track_data in tracker_ref.tracks.items():
                face_uuid = track_data.get("face_uuid")
                # We need an embedding stored in the track data for memory check
                t_emb = track_data.get("embedding")
                
                if face_uuid and t_emb is not None:
                    # similarity = 1 - cosine(a, b)
                    sim = 1 - cosine(embedding, t_emb)
                    if sim > best_score:
                        best_score = sim
                        best_match = face_uuid

        # Step 3: Check database (known_faces cache)
        # Only check DB if memory check didn't give a very strong match
        # though user says STRICT check ALL. Let's find global best.
        for known in self.known_faces:
            k_emb = known["embedding"]
            # Ensure known embedding is also normalized
            # (it should be persistent, but we check to be safe)
            sim = 1 - cosine(embedding, k_emb)
            
            if sim > best_score:
                best_score = sim
                best_match = known["face_uuid"]
        
        # Match threshold 0.30
        if best_score >= 0.30:
            return (best_match, best_score)
        
        return (None, best_score)

    def register_new_face(self, embedding: np.ndarray, timestamp: str) -> str:
        """
        L2 normalize and register new face in DB and cache.
        """
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        face_uuid = str(uuid.uuid4())
        database.register_face(self.conn, face_uuid, embedding, timestamp)
        
        self.known_faces.append({
            "face_uuid": face_uuid,
            "embedding": embedding
        })
        return face_uuid

    def reload_embeddings(self) -> None:
        """
        Re-fetch all embeddings from DB.
        """
        self.known_faces = database.get_all_embeddings(self.conn)
