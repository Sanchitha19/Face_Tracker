import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import base64
import json
import cv2
import time
from src.config_loader import get_config

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="FACETRACK PRO",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- CONFIG HELPER ---
def update_config(key_path: list, value):
    with open("config.json", "r") as f:
        cfg = json.load(f)
    ref = cfg
    for key in key_path[:-1]:
        ref = ref[key]
    ref[key_path[-1]] = value
    with open("config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    st.cache_resource.clear() # Force config reload

# --- CUSTOM CSS & FONTS ---
st.markdown("""
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400;1,700&family=Syne:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    
    <style>
    /* Global Styles */
    [data-testid="stAppViewContainer"] {
        background-color: #080c10;
        color: #e2e8f0;
        font-family: 'Syne', sans-serif;
    }
    
    [data-testid="stSidebar"] {
        background-color: #0d1117;
        border-right: 1px solid #1e2d3d;
    }
    
    h1, h2, h3, .mono {
        font-family: 'Space Mono', monospace !important;
    }
    
    header, footer, [data-testid="stToolbar"] {
        visibility: hidden;
    }
    
    .header-container {
        padding: 1.5rem 0;
        border-bottom: 1px solid #1e2d3d;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 15px;
    }
    
    .pulse-dot {
        width: 12px;
        height: 12px;
        background-color: #00ff88;
        border-radius: 50%;
        box-shadow: 0 0 10px #00ff88;
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 255, 136, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 10px rgba(0, 255, 136, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(0, 255, 136, 0); }
    }
    
    .title-text {
        color: #00ff88;
        font-size: 2.5rem;
        font-weight: 700;
        letter-spacing: -1px;
        text-shadow: 0 0 20px rgba(0, 255, 136, 0.3);
        margin: 0;
    }
    
    .subtitle-text {
        color: #4a6080;
        font-size: 0.9rem;
        margin-top: -5px;
    }

    .status-bar {
        display: flex;
        background-color: #0d1117;
        border: 1px solid #1e2d3d;
        border-radius: 6px;
        padding: 12px 20px;
        margin-top: 15px;
        font-family: 'Space Mono', monospace;
        font-size: 0.85rem;
        align-items: center;
        width: 100%;
        justify-content: space-between;
    }
    
    .status-pill {
        padding: 4px 12px;
        border-radius: 4px;
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    /* KPI Cards */
    .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin-bottom: 2rem; }
    .kpi-card { background-color: #0d1117; padding: 1.5rem; border-radius: 8px; border: 1px solid #1e2d3d; position: relative; }
    .kpi-label { color: #4a6080; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 1px; }
    .kpi-value { color: #e2e8f0; font-size: 2rem; font-weight: 700; margin: 10px 0; font-family: 'Space Mono', monospace; }
    .kpi-unique { border-top: 3px solid #00ff88; }
    .kpi-today { border-top: 3px solid #0ea5e9; }
    .kpi-entries { border-top: 3px solid #f59e0b; }
    .kpi-exits { border-top: 3px solid #ef4444; }

    /* Custom Tables */
    .styled-table { width: 100%; border-collapse: collapse; font-size: 0.9rem; background-color: #0d1117; border-radius: 8px; overflow: hidden; }
    .styled-table thead tr { background-color: #161b22; color: #4a6080; text-align: left; }
    .styled-table th, .styled-table td { padding: 12px 15px; border-bottom: 1px solid #1e2d3d; }
    .styled-table tbody tr:hover { background-color: #1c2128; }
    
    .badge-entry { background-color: rgba(0, 255, 136, 0.1); color: #00ff88; border: 1px solid #00ff88; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; }
    .badge-exit { background-color: rgba(239, 68, 68, 0.1); color: #ef4444; border: 1px solid #ef4444; padding: 2px 6px; border-radius: 4px; font-size: 0.7rem; }

    /* Gallery */
    .gallery-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 15px; }
    .face-card { background-color: #0d1117; border: 1px solid #1e2d3d; border-radius: 6px; padding: 8px; transition: all 0.3s ease; }
    .face-card:hover { transform: translateY(-5px); border-color: #00ff88; box-shadow: 0 0 15px rgba(0, 255, 136, 0.2); }
    .face-card-exit:hover { border-color: #ef4444; box-shadow: 0 0 15px rgba(239, 68, 68, 0.2); }
    .face-img { width: 100%; height: 120px; object-fit: cover; border-radius: 4px; margin-bottom: 5px; }
    .face-label { font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #4a6080; text-align: center; }

    .log-box { background-color: #0d1117; padding: 1rem; border-radius: 8px; border: 1px solid #1e2d3d; height: 300px; overflow-y: scroll; font-family: 'Space Mono', monospace; font-size: 0.8rem; line-height: 1.5; }
    .log-entry { color: #00ff88; }
    .log-exit { color: #ef4444; }
    .log-muted { color: #4a6080; }
    </style>
""", unsafe_allow_html=True)

# --- UTILS ---
@st.cache_resource
def get_db_connection():
    try:
        config = get_config()
        db_path = config["database"]["path"]
        if not os.path.exists(os.path.dirname(db_path)): os.makedirs(os.path.dirname(db_path), exist_ok=True)
        return sqlite3.connect(db_path, check_same_thread=False)
    except: return None

def fetch_data(query, params=None):
    conn = get_db_connection()
    if conn is None: return pd.DataFrame()
    return pd.read_sql_query(query, conn, params=params)

def get_base64_img(file_path):
    try:
        with open(file_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except: return None

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<h2 class='mono' style='color:#0ea5e9'>CONTROL PANEL</h2>", unsafe_allow_html=True)
    live_refresh = st.toggle("Live Refresh Feed", value=True)
    refresh_interval = st.slider("Refresh Interval (s)", 2, 30, 5)
    st.divider()
    current_date = st.date_input("Filter by Date", datetime.now())
    date_str = current_date.strftime("%Y-%m-%d")
    max_events = st.slider("Max Events to Display", 10, 100, 30)
    show_exits = st.toggle("Show Exit Gallery", value=False)
    
    # SETTINGS EXPANDER
    st.divider()
    with st.expander("⚙️ Detection Settings"):
        cfg = get_config()
        
        skip = st.slider("Skip Frames", 1, 10, cfg.get("skip_frames", 3))
        if skip != cfg.get("skip_frames"): update_config(["skip_frames"], skip)
        
        conf = st.slider("Confidence Threshold", 0.1, 0.9, cfg["detection"].get("confidence_threshold", 0.5), 0.05)
        if conf != cfg["detection"].get("confidence_threshold"): update_config(["detection", "confidence_threshold"], conf)
        
        sim = st.slider("Similarity Threshold", 0.1, 0.9, cfg["recognition"].get("similarity_threshold", 0.3), 0.05)
        if sim != cfg["recognition"].get("similarity_threshold"): update_config(["recognition", "similarity_threshold"], sim)
        
        min_sz = st.slider("Min Face Size (px)", 20, 200, cfg["recognition"].get("min_face_size", 40))
        if min_sz != cfg["recognition"].get("min_face_size"): update_config(["recognition", "min_face_size"], min_sz)
        
        max_dis = st.slider("Max Disappeared (frames)", 10, 120, cfg["tracking"].get("max_disappeared", 60))
        if max_dis != cfg["tracking"].get("max_disappeared"): update_config(["tracking", "max_disappeared"], max_dis)
        
        if st.button("💾 Save Settings", use_container_width=True):
            st.success("✅ Settings saved to config.json")

# --- HEADER ---
now_str = datetime.now().strftime("%I:%M:%S %p")
st.markdown(f"""
    <div class="header-container">
        <div class="pulse-dot"></div>
        <div>
            <h1 class="title-text mono">FACETRACK PRO</h1>
            <div class="subtitle-text">Intelligent Visitor Intelligence System · {datetime.now().strftime("%Y-%m-%d")} · {now_str}</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# --- SECTION: SOURCE CONFIGURATION PANEL ---
st.markdown("<h3 class='mono' style='margin-bottom:10px'>📁 SOURCE CONFIGURATION</h3>", unsafe_allow_html=True)
t1, t2 = st.tabs(["📁 Video File", "📡 RTSP Stream"])

with t1:
    uploaded_file = st.file_uploader("Upload video file", type=["mp4", "avi", "mov", "mkv"])
    local_path = st.text_input("Or enter local file path", placeholder="C:/videos/sample.mp4")
    
    if uploaded_file:
        with open("uploaded_video.mp4", "wb") as f: f.write(uploaded_file.getbuffer())
        update_config(["video_source"], "uploaded_video.mp4")
        update_config(["use_rtsp"], False)
        st.success("✅ Video file ready: uploaded_video.mp4")
    
    if local_path and st.button("Set Local Path", key="set_local"):
        if os.path.exists(local_path):
            update_config(["video_source"], local_path)
            update_config(["use_rtsp"], False)
            st.success(f"✅ Source set to: {local_path}")
        else:
            st.error("❌ File not found at provided path.")

with t2:
    rtsp_url = st.text_input("RTSP Stream URL", placeholder="rtsp://192.168.1.100:554/stream")
    u_col, p_col = st.columns(2)
    user = u_col.text_input("Username (optional)")
    pwd = p_col.text_input("Password (optional)", type="password")
    
    test_conn = st.toggle("Test connection before saving", value=True)
    
    if st.button("💾 Save RTSP Config", use_container_width=True):
        final_url = rtsp_url
        if user and pwd and "@" not in rtsp_url:
            clean_url = rtsp_url.replace("rtsp://", "")
            final_url = f"rtsp://{user}:{pwd}@{clean_url}"
        
        if test_conn:
            with st.spinner("Testing RTSP connection..."):
                cap = cv2.VideoCapture(final_url)
                if cap.isOpened():
                    st.success("✅ Connection successful")
                    cap.release()
                else:
                    st.error("❌ Cannot connect to stream. Check URL and network.")
                    st.stop()
        
        update_config(["rtsp_url"], final_url)
        update_config(["use_rtsp"], True)
        st.success("✅ RTSP stream configured")

# SOURCE STATUS BAR
cfg = get_config()
st_type = "📁 VIDEO FILE" if not cfg.get("use_rtsp") else "📡 RTSP LIVE"
st_color = "#0ea5e9" if not cfg.get("use_rtsp") else "#00ff88"
st_path = cfg.get("rtsp_url") if cfg.get("use_rtsp") else cfg.get("video_source")
m_time = time.strftime('%H:%M:%S', time.localtime(os.path.getmtime("config.json")))

st.markdown(f"""
    <div class="status-bar">
        <div style="display:flex; align-items:center; gap:10px">
            <span class="status-pill" style="background:{st_color}44; color:{st_color}; border:1px solid {st_color}">SOURCE: {st_type}</span>
            <span style="color:#e2e8f0">{st_path}</span>
        </div>
        <div style="color:#4a6080">Last Updated: {m_time}</div>
    </div>
""", unsafe_allow_html=True)

# START / STOP CONTROLS
st.markdown("<br>", unsafe_allow_html=True)
c1, c2, c3 = st.columns([1, 1, 2])
if c1.button("▶ START TRACKER", use_container_width=True, type="primary"):
    with open("tracker_command.txt", "w") as f: f.write("start")
    st.toast("Tracker command: START sent")
if c2.button("⏹ STOP TRACKER", use_container_width=True):
    with open("tracker_command.txt", "w") as f: f.write("stop")
    st.toast("Tracker command: STOP sent")
c3.info("Run in terminal: `python main.py`")

# --- DATA SECTION ---
conn = get_db_connection()
if conn is None:
    st.error("⚠️ Database initializing... Run `main.py` if this persists.")
    st.stop()

# KPIs
unique_visitors = fetch_data("SELECT COUNT(*) as count FROM faces").iloc[0]['count']
today_visitors = fetch_data("SELECT COUNT(DISTINCT face_uuid) as count FROM events WHERE date(timestamp) = ?", (date_str,)).iloc[0]['count']
total_entries = fetch_data("SELECT COUNT(*) as count FROM events WHERE event_type = 'entry'").iloc[0]['count']
total_exits = fetch_data("SELECT COUNT(*) as count FROM events WHERE event_type = 'exit'").iloc[0]['count']

st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card kpi-unique"><div class="kpi-label">Unique Visitors</div><div class="kpi-value">{unique_visitors}</div><div style="color:#00ff88;font-size:0.7rem">All-time</div></div>
        <div class="kpi-card kpi-today"><div class="kpi-label">Today's Visitors</div><div class="kpi-value">{today_visitors}</div><div style="color:#0ea5e9;font-size:0.7rem">{date_str}</div></div>
        <div class="kpi-card kpi-entries"><div class="kpi-label">Total Entries</div><div class="kpi-value">{total_entries}</div><div style="color:#f59e0b;font-size:0.7rem">Events</div></div>
        <div class="kpi-card kpi-exits"><div class="kpi-label">Total Exits</div><div class="kpi-value">{total_exits}</div><div style="color:#ef4444;font-size:0.7rem">Logged</div></div>
    </div>
""", unsafe_allow_html=True)

# CHARTS
gl1, gl2 = st.columns(2)
with gl1:
    df_daily = fetch_data("SELECT date(timestamp) as day, COUNT(DISTINCT face_uuid) as count FROM events GROUP BY day ORDER BY day DESC LIMIT 7")
    fig1 = px.bar(df_daily, x='day', y='count', color_discrete_sequence=['#00ff88'], title="Daily Unique Visitors")
    fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family='Space Mono', font_color='#e2e8f0', xaxis=dict(showgrid=False), yaxis=dict(gridcolor='#1e2d3d'))
    st.plotly_chart(fig1, use_container_width=True)
with gl2:
    df_hourly = fetch_data("SELECT strftime('%H:00', timestamp) as hour, event_type, COUNT(*) as count FROM events WHERE date(timestamp) = ? GROUP BY hour, event_type", (date_str,))
    fig2 = px.area(df_hourly, x='hour', y='count', color='event_type', color_discrete_map={'entry':'#f59e0b', 'exit':'#ef4444'}, title="Hourly Activity")
    fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_family='Space Mono', font_color='#e2e8f0', xaxis=dict(showgrid=False), yaxis=dict(gridcolor='#1e2d3d'))
    st.plotly_chart(fig2, use_container_width=True)

# FEED & TABLE
f_col, v_col = st.columns([3, 2])
with f_col:
    st.markdown("<h4 class='mono'>LIVE EVENT FEED</h4>", unsafe_allow_html=True)
    df_events = fetch_data("SELECT face_uuid, event_type, timestamp, track_id FROM events ORDER BY timestamp DESC LIMIT ?", (max_events,))
    if not df_events.empty:
        html = "<table class='styled-table'><thead><tr><th>Type</th><th>UUID</th><th>Track</th><th>Time</th></tr></thead><tbody>"
        for _, r in df_events.iterrows():
            badge = "badge-entry" if r['event_type'] == 'entry' else "badge-exit"
            html += f"<tr><td><span class='{badge}'>{r['event_type']}</span></td><td style='color:#0ea5e9;'>{r['face_uuid'][:10]}...</td><td>{r['track_id']}</td><td>{r['timestamp'].split(' ')[1]}</td></tr>"
        st.markdown(html + "</tbody></table>", unsafe_allow_html=True)
with v_col:
    st.markdown("<h4 class='mono'>VISITOR RANKING</h4>", unsafe_allow_html=True)
    df_rank = fetch_data("SELECT face_uuid, visit_count, last_seen FROM faces ORDER BY visit_count DESC LIMIT 10")
    if not df_rank.empty:
        html = "<table class='styled-table'><thead><tr><th>UUID</th><th>Visits</th><th>Last Seen</th></tr></thead><tbody>"
        for _, r in df_rank.iterrows():
            html += f"<tr><td style='color:#0ea5e9;'>{r['face_uuid'][:8]}</td><td style='color:#f59e0b;font-weight:700'>{r['visit_count']}</td><td>{r['last_seen'].split(' ')[1]}</td></tr>"
        st.markdown(html + "</tbody></table>", unsafe_allow_html=True)

# GALLERIES
st.markdown("<h3 class='mono' style='border-bottom:1px solid #1e2d3d; padding-bottom:10px'>ENTRY GALLERY</h3>", unsafe_allow_html=True)
df_imgs = fetch_data("SELECT face_uuid, image_path, timestamp FROM events WHERE event_type = 'entry' AND date(timestamp) = ? ORDER BY timestamp DESC LIMIT 12", (date_str,))
if not df_imgs.empty:
    cols = st.columns(6)
    for i, (_, row) in enumerate(df_imgs.iterrows()):
        b64 = get_base64_img(row['image_path'])
        if b64: cols[i % 6].markdown(f'<div class="face-card"><img src="data:image/jpeg;base64,{b64}" class="face-img"><div class="face-label">{row["face_uuid"][:8]}</div></div>', unsafe_allow_html=True)
else: st.info("No images for this date.")

if show_exits:
    st.markdown("<h3 class='mono' style='border-bottom:1px solid #1e2d3d; padding-bottom:10px; color:#ef4444'>EXIT GALLERY</h3>", unsafe_allow_html=True)
    df_exits = fetch_data("SELECT face_uuid, image_path, timestamp FROM events WHERE event_type = 'exit' AND date(timestamp) = ? ORDER BY timestamp DESC LIMIT 12", (date_str,))
    if not df_exits.empty:
        cols = st.columns(6)
        for i, (_, row) in enumerate(df_exits.iterrows()):
            b64 = get_base64_img(row['image_path'])
            if b64: cols[i % 6].markdown(f'<div class="face-card face-card-exit"><img src="data:image/jpeg;base64,{b64}" class="face-img"><div class="face-label">{row["face_uuid"][:8]}</div></div>', unsafe_allow_html=True)

# LOGS
st.markdown("<h4 class='mono'>SYSTEM LOGS</h4>", unsafe_allow_html=True)
log_p = cfg["logging"]["log_file"]
if os.path.exists(log_p):
    with open(log_p, "r") as f: last_l = f.readlines()[-40:][::-1]
    log_h = "<div class='log-box'>"
    for l in last_l:
        style = "log-muted"
        if "[ENTRY]" in l: style = "log-entry"
        elif "[EXIT]" in l: style = "log-exit"
        log_h += f"<div class='{style}'>{l.strip()}</div>"
    st.markdown(log_h + "</div>", unsafe_allow_html=True)

# FOOTER
st.markdown(f"<div style='border-top:1px solid #1e2d3d; padding:20px 0; text-align:center; color:#4a6080; font-family:Space Mono; font-size:0.8rem;'>FACETRACK PRO · SYSTEM UP-TIME: {now_str}</div>", unsafe_allow_html=True)

# AUTO REFRESH
if live_refresh:
    time.sleep(refresh_interval)
    st.rerun()
