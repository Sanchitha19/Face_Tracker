import sqlite3
import numpy as np
import os

def init_db(db_path: str) -> sqlite3.Connection:
    """
    Creates the DB file and all 3 tables if they don't exist.
    Returns a persistent connection.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()

    # Table: faces
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS faces (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        face_uuid   TEXT UNIQUE NOT NULL,
        first_seen  TEXT NOT NULL,
        last_seen   TEXT NOT NULL,
        visit_count INTEGER DEFAULT 1,
        embedding   BLOB NOT NULL
    )
    """)

    # Table: events
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS events (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        face_uuid   TEXT NOT NULL,
        event_type  TEXT NOT NULL CHECK(event_type IN ('entry', 'exit')),
        timestamp   TEXT NOT NULL,
        image_path  TEXT,
        track_id    INTEGER,
        FOREIGN KEY (face_uuid) REFERENCES faces(face_uuid)
    )
    """)

    # Table: visitor_summary
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS visitor_summary (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        session_date    TEXT NOT NULL,
        unique_visitors INTEGER DEFAULT 0,
        total_events    INTEGER DEFAULT 0,
        session_start   TEXT,
        session_end     TEXT
    )
    """)

    conn.commit()
    return conn

def register_face(conn, face_uuid: str, embedding: np.ndarray, timestamp: str) -> None:
    """
    Inserts a new face with its serialized embedding.
    """
    cursor = conn.cursor()
    embedding_blob = embedding.tobytes()
    cursor.execute("""
    INSERT INTO faces (face_uuid, first_seen, last_seen, visit_count, embedding)
    VALUES (?, ?, ?, 1, ?)
    """, (face_uuid, timestamp, timestamp, embedding_blob))
    conn.commit()

def get_all_embeddings(conn) -> list:
    """
    Returns list of {face_uuid, embedding (as np.ndarray)} for all registered faces.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT face_uuid, embedding FROM faces")
    rows = cursor.fetchall()
    
    known_faces = []
    for row in rows:
        face_uuid, embedding_blob = row
        embedding = np.frombuffer(embedding_blob, dtype=np.float32)
        known_faces.append({"face_uuid": face_uuid, "embedding": embedding})
    
    return known_faces

def update_last_seen(conn, face_uuid: str, timestamp: str) -> None:
    """
    Updates last_seen and increments visit_count.
    """
    cursor = conn.cursor()
    cursor.execute("""
    UPDATE faces 
    SET last_seen = ?, visit_count = visit_count + 1
    WHERE face_uuid = ?
    """, (timestamp, face_uuid))
    conn.commit()

def log_event(conn, face_uuid: str, event_type: str, timestamp: str,
             image_path: str, track_id: int) -> None:
    """
    Inserts one row into events table.
    """
    cursor = conn.cursor()
    cursor.execute("""
    INSERT INTO events (face_uuid, event_type, timestamp, image_path, track_id)
    VALUES (?, ?, ?, ?, ?)
    """, (face_uuid, event_type, timestamp, image_path, track_id))
    conn.commit()

def get_unique_visitor_count(conn) -> int:
    """
    Returns COUNT(DISTINCT face_uuid) from the faces table.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(DISTINCT face_uuid) FROM faces")
    count = cursor.fetchone()[0]
    return count

def upsert_visitor_summary(conn, session_date: str, unique_visitors: int,
                           total_events: int, session_start: str,
                           session_end: str) -> None:
    """
    Insert or replace a summary row for the given date.
    """
    cursor = conn.cursor()
    # Check if a summary exists for this date
    cursor.execute("SELECT id FROM visitor_summary WHERE session_date = ?", (session_date,))
    row = cursor.fetchone()
    
    if row:
        # Update existing
        cursor.execute("""
        UPDATE visitor_summary
        SET unique_visitors = ?, total_events = ?, 
            session_start = ?, session_end = ?
        WHERE session_date = ?
        """, (unique_visitors, total_events, session_start, session_end, session_date))
    else:
        # Insert new
        cursor.execute("""
        INSERT INTO visitor_summary (session_date, unique_visitors, total_events, session_start, session_end)
        VALUES (?, ?, ?, ?, ?)
        """, (session_date, unique_visitors, total_events, session_start, session_end))
    
    conn.commit()
def get_visit_count(conn, face_uuid: str) -> int:
    """
    Returns visit_count for a face_uuid.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT visit_count FROM faces WHERE face_uuid = ?", (face_uuid,))
    res = cursor.fetchone()
    return res[0] if res else 0

def get_all_faces_summary(conn) -> list:
    """
    Returns list of (face_uuid, visit_count, first_seen, last_seen).
    """
    cursor = conn.cursor()
    cursor.execute("SELECT face_uuid, visit_count, first_seen, last_seen FROM faces ORDER BY first_seen ASC")
    return cursor.fetchall()
