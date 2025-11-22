"""
app.py - Violence Detection (Updated, session-state smoothing)
- Non-violence = 0.7 * raw_non + 0.3 * (1 - raw_viol)
- EMA smoothing alpha stored in st.session_state (no global modification)
- Robust checkpoint loader using safe_globals then fallback
- Autoplay annotated frames, right panel live metrics, CSV export
- Demo video path uses uploaded file: /mnt/data/fighting2.mp4
"""

import os
import time
import tempfile
import warnings
import logging
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch import serialization
from ultralytics import YOLO
import mediapipe as mp
import streamlit as st
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
import matplotlib.pyplot as plt
import pandas as pd

# -------------------------
# Config & defaults
# -------------------------
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('mediapipe').setLevel(logging.ERROR)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

DEFAULT_IMG_SIZE = 320
DEFAULT_SEQUENCE_LENGTH = 24
DEFAULT_SKIP_FRAMES = 4
DEFAULT_PROXIMITY_THRESHOLD = 0.10
DEFAULT_SMOOTH_ALPHA = 0.65

st.set_page_config(page_title="Violence Detection", page_icon="⚠️", layout="wide")

st.markdown("""
<style>
  .main-header { font-size: 34px; color: #FF4B4B; font-weight: 700; margin-bottom: 6px; }
  .sub-header { color:#666; margin-bottom:12px; }
  .violence-alert { background:#FF4B4B; color:white; padding:8px; border-radius:8px; text-align:center; font-weight:700; }
  .safe-alert { background:#00CC66; color:white; padding:8px; border-radius:8px; text-align:center; font-weight:700; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# Model classes
# -------------------------
class PoseGNN(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, output_dim=128, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GATConv(input_dim, hidden_dim, heads=4, concat=True))
        self.bns.append(nn.BatchNorm1d(hidden_dim * 4))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_dim * 4, hidden_dim, heads=4, concat=True))
            self.bns.append(nn.BatchNorm1d(hidden_dim * 4))
        self.convs.append(GATConv(hidden_dim * 4, output_dim, heads=1, concat=False))
        self.bns.append(nn.BatchNorm1d(output_dim))

    def forward(self, x, edge_index, batch=None):
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            x = conv(x, edge_index)
            x = bn(x)
            if i < len(self.convs) - 1:
                x = torch.relu(x)
        return x

class GNNViolenceDetector(nn.Module):
    def __init__(self, yolo_dim=20, proj_dim=128, gnn_hidden=128, lstm_hidden=256,
                 gnn_layers=3, lstm_layers=2, dropout=0.4):
        super().__init__()
        self.yolo_projection = nn.Linear(yolo_dim, proj_dim)
        self.gnn = PoseGNN(4, gnn_hidden, gnn_hidden, gnn_layers)
        self.gnn_projection = nn.Linear(gnn_hidden, proj_dim)
        self.lstm = nn.LSTM(proj_dim * 2, lstm_hidden, lstm_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        lstm_out = lstm_hidden * 2
        self.attention = nn.Linear(lstm_out, 1)
        self.fc1 = nn.Linear(lstm_out, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, yolo_seq, graph_list, keypoint_seq=None):
        B, T, _ = yolo_seq.shape
        y_proj = self.yolo_projection(yolo_seq)  # (B,T,proj)
        g_feats = []
        for g in graph_list:
            g_out = self.gnn(g.x, g.edge_index)
            g_feats.append(g_out.mean(dim=0))
        g_stack = torch.stack(g_feats).view(1, T, -1).to(y_proj.device)
        g_proj = self.gnn_projection(g_stack)
        combined = torch.cat([y_proj, g_proj], dim=-1)
        lstm_out, _ = self.lstm(combined)
        attn_w = torch.softmax(self.attention(lstm_out), dim=1)
        attended = (lstm_out * attn_w).sum(dim=1)
        x = torch.relu(self.bn1(self.fc1(attended)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        logits = self.fc4(x)
        return logits, attn_w

# -------------------------
# Feature extractor
# -------------------------
class HybridFeatureExtractor:
    def __init__(self, yolo_model_path='yolov8n.pt', img_size=DEFAULT_IMG_SIZE):
        self.yolo = YOLO(yolo_model_path)
        self.yolo.overrides['verbose'] = False
        self.img_size = img_size
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True, model_complexity=1, min_detection_confidence=0.5)
        self.skeleton_edges = [
            (0,1),(0,4),(1,2),(2,3),(3,7),(4,5),(5,6),(6,8),
            (9,10),(11,12),(11,13),(11,23),(12,14),(12,24),(13,15),(14,16),
            (15,17),(15,19),(15,21),(16,18),(16,20),(16,22),(17,19),(18,20),
            (23,24),(23,25),(24,26),(25,27),(26,28),(27,29),(27,31),(28,30),(28,32)
        ]
        try:
            self.edge_index = torch.tensor(self.skeleton_edges, dtype=torch.long).t()
        except Exception:
            self.edge_index = torch.zeros((2,0), dtype=torch.long)

    def extract_frame_features(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        results = self.yolo(frame_bgr, imgsz=self.img_size, verbose=False)[0]
        person_boxes, weapon_boxes = [], []
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
            bbox = [x1/w, y1/h, x2/w, y2/h, conf]
            if cls == 0:
                person_boxes.append(bbox)
            elif cls in [39,40,41,42,43]:
                weapon_boxes.append(bbox)
        yolo_feat = np.zeros(20, dtype=np.float32)
        for i, b in enumerate(person_boxes[:2]):
            yolo_feat[i*5:(i+1)*5] = b
        for i, b in enumerate(weapon_boxes[:2]):
            yolo_feat[10+i*5:10+(i+1)*5] = b
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        keypoints_flat = np.zeros(99, dtype=np.float32)
        keypoints_2d = []
        if res.pose_landmarks:
            for i, lm in enumerate(res.pose_landmarks.landmark):
                keypoints_flat[i*3:(i+1)*3] = [lm.x, lm.y, lm.visibility]
                keypoints_2d.append([lm.x, lm.y, lm.visibility])
        if len(keypoints_2d) == 33:
            node_feats = [[kp[0], kp[1], kp[2], 1.0] for kp in keypoints_2d]
            x = torch.tensor(node_feats, dtype=torch.float32)
            graph = Data(x=x, edge_index=self.edge_index)
        else:
            graph = Data(x=torch.zeros((33,4), dtype=torch.float32), edge_index=self.edge_index)
        return {
            'yolo': yolo_feat,
            'keypoints': keypoints_flat,
            'graph': graph,
            'person_boxes': person_boxes,
            'weapon_boxes': weapon_boxes,
            'pose_landmarks': res.pose_landmarks if res.pose_landmarks else None
        }

# -------------------------
# Drawing helpers
# -------------------------
def draw_detections(frame, features, show_keypoints=True, show_boxes=True):
    annotated = frame.copy()
    h,w = annotated.shape[:2]
    if show_boxes:
        for b in features.get('person_boxes', []):
            x1,y1,x2,y2,conf = b
            x1,y1,x2,y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,255,0), 2)
            cv2.putText(annotated, f"P {conf:.2f}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        for b in features.get('weapon_boxes', []):
            x1,y1,x2,y2,conf = b
            x1,y1,x2,y2 = int(x1*w), int(y1*h), int(x2*w), int(y2*h)
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,0,255), 2)
            cv2.putText(annotated, f"W {conf:.2f}", (x1, y1-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
    if show_keypoints and features.get('pose_landmarks') is not None:
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        mp_drawing.draw_landmarks(
            annotated,
            features['pose_landmarks'],
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,255), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255,255,0), thickness=2)
        )
    return annotated

def draw_top_banner(frame_bgr, is_violence, violence_prob, non_violence_prob):
    overlay = frame_bgr.copy()
    h,w = frame_bgr.shape[:2]
    bh = max(56, int(h * 0.12))
    color = (0,0,255) if is_violence else (0,180,0)
    cv2.rectangle(overlay, (0,0), (w, bh), color, -1)
    cv2.addWeighted(overlay, 0.45, frame_bgr, 0.55, 0, frame_bgr)
    display_prob = violence_prob if is_violence else non_violence_prob
    # Clean labels without any special characters or emojis
    label = "VIOLENCE DETECTED" if is_violence else "NON-VIOLENCE"
    txt = "{}: {:.1f}%".format(label, display_prob*100)
    cv2.putText(frame_bgr, txt, (18, int(bh*0.6)), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255,255,255), 2, cv2.LINE_AA)
    return frame_bgr

# -------------------------
# Robust checkpoint loader
# -------------------------
def load_model_robust(model_path, model_constructor):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_constructor()
    checkpoint = None
    try:
        with serialization.safe_globals(['numpy.core.multiarray.scalar']):
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except Exception as e1:
        logging.warning("safe_globals load failed: %s", e1)
        try:
            checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        except Exception as e2:
            logging.warning("weights_only=True failed: %s", e2)
            st.warning("Attempting fallback load with weights_only=False. Only use this if you TRUST the checkpoint.")
            checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state = checkpoint['state_dict']
    else:
        state = checkpoint
    try:
        model.load_state_dict(state)
    except Exception as e:
        logging.warning("Strict load failed: %s. Retrying with strict=False", e)
        model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return {'model': model, 'device': device, 'violence_idx': 1, 'non_idx': 0}

# -------------------------
# Utility functions
# -------------------------
def ema_update(prev, current, alpha):
    if prev is None:
        return current
    return alpha * prev + (1.0 - alpha) * current

def centers_from_boxes(boxes):
    centers = []
    for b in boxes:
        cx = (b[0] + b[2]) / 2.0
        cy = (b[1] + b[3]) / 2.0
        centers.append((cx, cy))
    return centers

def min_pairwise_distance(boxes):
    if len(boxes) < 2:
        return 1.0
    centers = centers_from_boxes(boxes)
    min_d = 1.0
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            d = np.hypot(dx, dy)
            if d < min_d:
                min_d = d
    return min_d

# -------------------------
# Prediction (streamlit-ready)
# -------------------------
def predict_video_frame_by_frame_streamlit(video_path, model_bundle, extractor,
                                          sequence_length=DEFAULT_SEQUENCE_LENGTH,
                                          skip_frames=DEFAULT_SKIP_FRAMES,
                                          enable_proximity_gating=False,
                                          proximity_threshold=DEFAULT_PROXIMITY_THRESHOLD,
                                          person_confidence_threshold=0.3):
    device = model_bundle['device']
    model = model_bundle['model']
    violence_idx = model_bundle.get('violence_idx', 1)
    non_idx = model_bundle.get('non_idx', 0)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Cannot open video: {video_path}")
        return [], [], [], []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    yolo_buffer = deque(maxlen=sequence_length)
    graph_buffer = deque(maxlen=sequence_length)
    keypoint_buffer = deque(maxlen=sequence_length)

    frame_results = []
    annotated_frames = []
    features_list = []

    prev_violence_ema = None
    prev_non_ema = None

    frame_id = 0
    processed = 0
    progress = st.progress(0.0)
    status = st.empty()

    # smoothing alpha from session_state (fallback to default)
    smooth_alpha = st.session_state.get('smooth_alpha', DEFAULT_SMOOTH_ALPHA)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % skip_frames != 0:
            frame_id += 1
            processed += 1
            if total_frames:
                progress.progress(min(processed / total_frames, 1.0))
            continue

        resized = cv2.resize(frame, (extractor.img_size, extractor.img_size))
        features = extractor.extract_frame_features(resized)

        # filter persons by confidence threshold
        persons = [p for p in features.get('person_boxes', []) if p[4] >= person_confidence_threshold]
        weapons = features.get('weapon_boxes', [])

        features['person_boxes'] = persons
        features_list.append(features)
        yolo_buffer.append(features['yolo'])
        graph_buffer.append(features['graph'])
        keypoint_buffer.append(features['keypoints'])

        violence_prob_raw = 0.0
        non_prob_raw = 1.0
        prediction = 0

        # No persons -> safe
        if len(persons) == 0:
            violence_prob_raw = 0.0
            non_prob_raw = 1.0
            prediction = 0
        elif len(yolo_buffer) == sequence_length:
            yolo_seq = torch.FloatTensor(np.array(list(yolo_buffer))).unsqueeze(0).to(device)
            kp_seq = torch.FloatTensor(np.array(list(keypoint_buffer))).unsqueeze(0).to(device)
            graph_list = []
            for g in list(graph_buffer):
                if hasattr(g, 'x'):
                    g.x = g.x.to(device)
                if hasattr(g, 'edge_index'):
                    g.edge_index = g.edge_index.to(device)
                graph_list.append(g)
            with torch.no_grad():
                out = model(yolo_seq, graph_list, kp_seq)
                logits = out[0] if isinstance(out, (list, tuple)) else out
                probs = torch.softmax(logits, dim=1)
                raw_non = float(probs[0, non_idx].item())
                raw_viol = float(probs[0, violence_idx].item())

                # Weighted hybrid rule (option 3)
                hybrid_non = 0.7 * raw_non + 0.3 * (1.0 - raw_viol)
                violence_prob_raw = raw_viol
                non_prob_raw = float(np.clip(hybrid_non, 0.0, 1.0))

                # optional proximity gating
                if enable_proximity_gating:
                    min_d = min_pairwise_distance(persons)
                    if (min_d >= proximity_threshold) and (len(weapons) == 0):
                        violence_prob_raw = 0.0
                        non_prob_raw = 1.0

                prediction = 1 if violence_prob_raw > 0.5 else 0
        else:
            # warm-up: use previous EMA or defaults
            violence_prob_raw = prev_violence_ema if prev_violence_ema is not None else 0.0
            non_prob_raw = prev_non_ema if prev_non_ema is not None else 1.0
            prediction = 1 if violence_prob_raw > 0.5 else 0

        # EMA smoothing using session state alpha
        prev_violence_ema = ema_update(prev_violence_ema, violence_prob_raw, smooth_alpha)
        prev_non_ema = ema_update(prev_non_ema, non_prob_raw, smooth_alpha)

        display_viol = float(prev_violence_ema)
        display_non = float(prev_non_ema)

        annotated = frame.copy()
        try:
            annotated = draw_detections(annotated, features, show_keypoints=True, show_boxes=True)
        except Exception:
            pass
        annotated = draw_top_banner(annotated, display_viol > 0.5, display_viol, display_non)

        frame_results.append({
            'frame': frame_id,
            'violence_prob': display_viol,
            'non_violence_prob': display_non,
            'prediction': 1 if display_viol > 0.5 else 0
        })
        annotated_frames.append(annotated)

        frame_id += 1
        processed += 1
        if total_frames:
            progress.progress(min(processed / total_frames, 1.0))
        status.text(f"Processed frames: {processed}/{total_frames}")

    cap.release()
    progress.empty()
    status.empty()
    playback_fps = max(1, int(round(fps / max(skip_frames, 1))))
    return frame_results, annotated_frames, features_list, playback_fps

# -------------------------
# UI & main
# -------------------------
def main():
    st.markdown("<div class='main-header'>⚠️ Violence Detection</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-header'>Weighted hybrid non-violence + EMA smoothing stored in session state.</div>", unsafe_allow_html=True)

    # initialize smoothing alpha in session state if missing
    if 'smooth_alpha' not in st.session_state:
        st.session_state['smooth_alpha'] = DEFAULT_SMOOTH_ALPHA

    left_col, right_col = st.columns([3, 1])

    with right_col:
        st.markdown("### Model / Config")
        st.write(f"- Image size: {DEFAULT_IMG_SIZE}×{DEFAULT_IMG_SIZE}")
        st.write(f"- Sequence length (default): {DEFAULT_SEQUENCE_LENGTH}")
        st.write(f"- Default skip frames: {DEFAULT_SKIP_FRAMES}")
        st.markdown("---")
        st.markdown("### Detected Objects")
        persons_placeholder = st.empty()
        weapons_placeholder = st.empty()
        st.markdown("**Person Detections:**")
        person_list_placeholder = st.empty()
        st.markdown("---")
        live_label = st.empty()
        live_bar = st.empty()
        live_metrics = st.empty()

    # sidebar
    st.sidebar.title("Settings")
    model_path = st.sidebar.text_input("Model Path", "gnn_yolov8n_violence_detection.pth")
    sequence_length = st.sidebar.slider("Sequence Length", 8, 48, DEFAULT_SEQUENCE_LENGTH, step=4)
    skip_frames = st.sidebar.slider("Frame Skip (process every Nth frame)", 1, 8, DEFAULT_SKIP_FRAMES)
    person_conf_thresh = st.sidebar.slider("Person confidence threshold (YOLO)", 0.1, 0.7, 0.3, step=0.05)

    # smoothing slider updates session state (no global)
    alpha_percent = st.sidebar.slider("EMA smoothing alpha (higher = smoother)", 50, 90, int(DEFAULT_SMOOTH_ALPHA * 100))
    st.session_state['smooth_alpha'] = alpha_percent / 100.0

    st.sidebar.markdown("---")
    enable_proximity = st.sidebar.checkbox("Enable proximity gating (override if persons far)", value=False)
    proximity_threshold = st.sidebar.slider("Proximity threshold (normalized)", 5, 40, int(DEFAULT_PROXIMITY_THRESHOLD*100)) / 100.0

    st.sidebar.markdown("---")
    st.sidebar.markdown("Hybrid non-violence rule (fixed):")
    st.sidebar.markdown("non = 0.7 * raw_non + 0.3 * (1 - raw_viol)")

    st.sidebar.markdown("---")
    st.sidebar.markdown("Upload a video:")
    uploaded_file = st.file_uploader("Upload Video", type=['mp4','avi','mov','mkv'])
    video_to_use = None
    if uploaded_file is not None:
        t = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        t.write(uploaded_file.read())
        video_to_use = t.name

    if video_to_use:
        if not os.path.exists(model_path):
            st.sidebar.error(f"Model file not found at: {model_path}")
            return

        with st.spinner("Loading model..."):
            try:
                model_bundle = load_model_robust(model_path, lambda: GNNViolenceDetector())
            except Exception as e:
                st.error(f"Model loading failed: {e}")
                return

        with st.spinner("Loading YOLO & MediaPipe..."):
            extractor = HybridFeatureExtractor('yolov8n.pt', img_size=DEFAULT_IMG_SIZE)

        with st.spinner("Processing video (frame-by-frame)..."):
            frame_results, annotated_frames, features_list, playback_fps = predict_video_frame_by_frame_streamlit(
                video_to_use, model_bundle, extractor,
                sequence_length=sequence_length,
                skip_frames=skip_frames,
                enable_proximity_gating=enable_proximity,
                proximity_threshold=proximity_threshold,
                person_confidence_threshold=person_conf_thresh
            )

        if uploaded_file is not None:
            try:
                os.unlink(video_to_use)
            except Exception:
                pass

        if not annotated_frames:
            st.error("No annotated frames produced; check model and extractor.")
            return

        play_box = left_col.empty()
        delay = 1.0 / max(1, playback_fps)

        violent_frames = sum(1 for r in frame_results if r['prediction'] == 1)
        avg_violence = np.mean([r['violence_prob'] for r in frame_results]) if frame_results else 0.0

        with right_col:
            st.markdown("### Video Statistics")
            st.write(f"Total frames: {len(annotated_frames)}")
            st.write(f"Violence frames: {violent_frames}")
            st.write(f"Avg violence prob: {avg_violence*100:.1f}%")

        for idx, frame_bgr in enumerate(annotated_frames):
            img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            play_box.image(img_rgb, use_container_width=True)

            r = frame_results[idx]
            viol = r.get('violence_prob', 0.0)
            nonv = r.get('non_violence_prob', 1.0)
            pred = r.get('prediction', 0)

            display_prob = viol if pred == 1 else nonv
            live_bar.progress(min(display_prob, 1.0))

            if pred == 1:
                live_label.markdown(f"<div class='violence-alert'>VIOLENCE<br>{display_prob*100:.1f}%</div>", unsafe_allow_html=True)
            else:
                live_label.markdown(f"<div class='safe-alert'>NON-VIOLENCE<br>{display_prob*100:.1f}%</div>", unsafe_allow_html=True)

            live_metrics.markdown(
                f"**Frame {idx+1}/{len(annotated_frames)}**  \n"
                f"Violence: {viol*100:.1f}%  \n"
                f"Non-Violence: {nonv*100:.1f}%  \n"
                f"Prediction: {'VIOLENCE' if pred==1 else 'NON-VIOLENCE'}"
            )

            feats = features_list[idx] if idx < len(features_list) else {}
            persons = feats.get('person_boxes', [])
            weapons = feats.get('weapon_boxes', [])
            persons_placeholder.markdown(f"**Persons:** {len(persons)}")
            weapons_placeholder.markdown(f"**Weapons:** {len(weapons)}")
            if persons:
                persons_sorted = sorted(persons, key=lambda x: x[4], reverse=True)
                txt = ""
                for i_p, p in enumerate(persons_sorted[:5]):
                    txt += f"• Person {i_p+1}: {p[4]*100:.0f}% confidence  \n"
                person_list_placeholder.markdown(txt)
            else:
                person_list_placeholder.markdown("• None")

            time.sleep(delay)

        play_box.image(cv2.cvtColor(annotated_frames[-1], cv2.COLOR_BGR2RGB), use_container_width=True)

        left_col.markdown("---")
        left_col.markdown("### 📊 Analysis Summary")
        col1, col2, col3 = left_col.columns(3)
        col1.metric("Total Frames", len(annotated_frames))
        violence_percentage = (violent_frames / len(annotated_frames)) * 100 if annotated_frames else 0.0
        col2.metric("Violence Frames", f"{violent_frames}", f"{violence_percentage:.1f}%")
        col3.metric("Overall Classification", "VIOLENCE" if violence_percentage > 50 else "SAFE", f"{avg_violence*100:.1f}% avg")

        frames_idx = [r['frame'] for r in frame_results]
        violence_probs = [r['violence_prob'] for r in frame_results]
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(frames_idx, violence_probs, linewidth=2, color='red', alpha=0.8)
        ax.axhline(0.5, color='black', linestyle='--', alpha=0.6)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Processed Frame Index")
        ax.set_ylabel("Violence Probability")
        left_col.pyplot(fig)

        results_df = pd.DataFrame(frame_results)
        csv = results_df.to_csv(index=False)
        left_col.download_button("Download Results (CSV)", data=csv, file_name="violence_results.csv", mime="text/csv")

        with left_col.expander("View results table"):
            st.dataframe(results_df, use_container_width=True)

    else:
        st.info("Please upload a video to start detection.")
        with right_col:
            st.markdown("### Detected Objects")
            st.markdown("**Persons:** 0")
            st.markdown("**Weapons:** 0")
            st.markdown("**Person Detections:**  \n• None")

if __name__ == "__main__":
    main()