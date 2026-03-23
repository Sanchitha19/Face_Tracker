"""
app.py — Streamlit dashboard for the Intelligent Face Tracker.
Run: streamlit run app.py
"""

import sqlite3
import os
from datetime import datetime, date
from pathlib import Path

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from PIL import Image

from src.config_loader import get_config

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Face Tracker Dashboard",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load config ─────────────────────────────────────────────────────────────────
cfg = get_config()
DB_PATH = cfg["database"]["path"]
LOG_PATH = cfg["logging"]["log_file"]
ENTRIES_DIR = Path("logs/entries")
EXITS_DIR = Path("logs/exits")


# ── DB helpers ──────────────────────────────────────────────────────────────────
@st.cache_resource
def get_connection():
    if not Path(DB_PATH).exists():
        st.error(
            f"Database not found at `{DB_PATH}`. "
            "Run `python main.py` first to start the tracker."
        )
        st.stop()
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def query(sql: str, params=()) -> pd.DataFrame:
    conn = get_connection()
    try:
        return pd.read_sql_query(sql, conn, params=params)
    except Exception as e:
        st.error(f"DB query error: {e}")
        return pd.DataFrame()


def scalar(sql: str, params=()) -> int:
    conn = get_connection()
    try:
        cur = conn.execute(sql, params)
        row = cur.fetchone()
        return row[0] if row and row[0] is not None else 0
    except Exception:
        return 0


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Controls")
    auto_refresh = st.toggle("Auto-refresh (5s)", value=False)
    selected_date = st.date_input("Filter by date", value=date.today())
    st.divider()
    st.caption(f"DB: `{DB_PATH}`")
    st.caption(f"Log: `{LOG_PATH}`")

if auto_refresh:
    st.empty()
    import time
    time.sleep(5)
    st.rerun()

# ── Header ───────────────────────────────────────────────────────────────────────
st.title("👁️ Intelligent Face Tracker")
st.caption(f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.divider()

# ── Top KPI cards ────────────────────────────────────────────────────────────────
total_unique   = scalar("SELECT COUNT(DISTINCT face_uuid) FROM faces")
total_entries  = scalar("SELECT COUNT(*) FROM events WHERE event_type='entry'")
total_exits    = scalar("SELECT COUNT(*) FROM events WHERE event_type='exit'")
today_str      = date.today().isoformat()
today_visitors = scalar(
    "SELECT COUNT(DISTINCT face_uuid) FROM events WHERE event_type='entry' AND timestamp LIKE ?",
    (f"{today_str}%",)
)

k1, k2, k3, k4 = st.columns(4)
k1.metric("🧑 Total Unique Visitors", total_unique)
k2.metric("📅 Today's Visitors", today_visitors)
k3.metric("🚪 Total Entries", total_entries)
k4.metric("🚶 Total Exits", total_exits)

st.divider()

# ── Charts row ───────────────────────────────────────────────────────────────────
col_left, col_right = st.columns(2)

# Visitors per day bar chart
with col_left:
    st.subheader("Visitors per day")
    df_daily = query("""
        SELECT DATE(timestamp) as day, COUNT(DISTINCT face_uuid) as visitors
        FROM events
        WHERE event_type = 'entry'
        GROUP BY day
        ORDER BY day
    """)
    if df_daily.empty:
        st.info("No entry data yet.")
    else:
        fig = px.bar(
            df_daily, x="day", y="visitors",
            labels={"day": "Date", "visitors": "Unique Visitors"},
            color_discrete_sequence=["#5DCAA5"]
        )
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=280)
        st.plotly_chart(fig, use_container_width=True)

# Entries vs exits over time
with col_right:
    st.subheader("Entries vs Exits — hourly")
    df_hourly = query("""
        SELECT
            strftime('%H:00', timestamp) as hour,
            event_type,
            COUNT(*) as count
        FROM events
        WHERE DATE(timestamp) = ?
        GROUP BY hour, event_type
        ORDER BY hour
    """, (selected_date.isoformat(),))
    if df_hourly.empty:
        st.info(f"No events on {selected_date}.")
    else:
        fig2 = px.line(
            df_hourly, x="hour", y="count", color="event_type",
            labels={"hour": "Hour", "count": "Events", "event_type": "Type"},
            color_discrete_map={"entry": "#1D9E75", "exit": "#D85A30"}
        )
        fig2.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=280)
        st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── Recent events table ──────────────────────────────────────────────────────────
st.subheader("📋 Recent Events")
df_events = query("""
    SELECT
        e.event_type,
        e.face_uuid,
        e.track_id,
        e.timestamp,
        e.image_path
    FROM events e
    ORDER BY e.timestamp DESC
    LIMIT 50
""")

if df_events.empty:
    st.info("No events logged yet. Start the tracker with `python main.py`.")
else:
    # Colour code entry vs exit
    def colour_type(val):
        color = "#1D9E75" if val == "entry" else "#D85A30"
        return f"color: {color}; font-weight: bold"

    styled = df_events.drop(columns=["image_path"]).style.applymap(
        colour_type, subset=["event_type"]
    )
    st.dataframe(styled, use_container_width=True, height=300)

st.divider()

# ── Face image gallery ───────────────────────────────────────────────────────────
st.subheader(f"🖼️ Entry Images — {selected_date}")

date_folder = ENTRIES_DIR / selected_date.isoformat()
if not date_folder.exists():
    st.info(f"No entry images found for {selected_date}.")
else:
    image_files = sorted(date_folder.glob("*.jpg"))
    if not image_files:
        st.info("No images in this folder.")
    else:
        cols = st.columns(6)
        for i, img_path in enumerate(image_files[:30]):
            with cols[i % 6]:
                try:
                    img = Image.open(img_path)
                    face_id = img_path.stem.split("_")[0][:8]
                    st.image(img, caption=face_id, use_column_width=True)
                except Exception:
                    st.warning("Unreadable image")

st.divider()

# ── Registered faces table ───────────────────────────────────────────────────────
st.subheader("🗂️ Registered Faces")
df_faces = query("""
    SELECT
        face_uuid,
        first_seen,
        last_seen,
        visit_count
    FROM faces
    ORDER BY first_seen DESC
""")

if df_faces.empty:
    st.info("No faces registered yet.")
else:
    st.dataframe(df_faces, use_container_width=True, height=250)

st.divider()

# ── System log viewer ────────────────────────────────────────────────────────────
st.subheader("📄 System Log (last 50 lines)")
if Path(LOG_PATH).exists():
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()
    last_lines = lines[-50:] if len(lines) > 50 else lines
    st.code("".join(last_lines), language="text")
else:
    st.info(f"Log file not found at `{LOG_PATH}`. Start the tracker first.")

# ── Footer ───────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with YOLOv8 · InsightFace · SQLite · Streamlit")
