#  Violence Detection Model (YOLO + GNN + LSTM)

A hybrid deep learning architecture designed for **real‑time human violence detection** using a powerful combination of:

* **YOLOv8** for person & object detection
* **MediaPipe Pose** for human keypoint extraction
* **GAT‑based Graph Neural Networks (GNNs)** for pose graph encoding
* **Bi‑LSTM + Attention** for temporal understanding
* **EMA smoothing** for stable, consistent predictions

This model is optimized for real‑time inference, smooth probability curves, and high interpretability.

---

##  Features

### ** Hybrid Multi‑Modal Architecture**

* Combines **bounding boxes**, **pose keypoints**, and **graph embeddings**.
* Learns both spatial pose relations and temporal motion patterns.

### **🎯Accurate Violence Classification**

* Frame‑wise classification into **Violence** or **Non‑Violence**.
* Optional **proximity gating** to prevent false positives.

### ** Stable Predictions**

* Uses **Exponential Moving Average (EMA)** smoothing.
* Weighted hybrid probability formula for non‑violence.

### ** Streamlit Interface**

* Real‑time frame display.
* Live probability bar.
* Detected persons, weapons, and pose graphs.
* Full video statistics + downloadable CSV.

---

## 🧠 Model Architecture

```
YOLOv8 (Bounding Boxes + Person/Weapon Info)
             │
             ▼
MediaPipe Pose → Pose Graph → GATConv Layers (GNN)
             │                       │
             └──────────┬────────────┘
                        ▼
             Projection + Fusion Layer
                        ▼
                Bi‑LSTM + Attention
                        ▼
                Fully Connected Layers
                        ▼
          Violence / Non‑Violence Output
```


---

##  Output Visualization

* Bounding boxes for persons & weapons
* Pose skeleton overlay
* Violence/Non‑Violence banner (emoji capable)
* Live probability graph
* Downloadable per‑frame CSV

---

##  Training Dataset

You can train using:

* Custom violence datasets
* CCTV footage
* Action recognition datasets
* Real Life Violence Dataset

Supports noisy datasets via:

* EMA smoothing
* Hybrid probability weighting

---

##  Example Predictions

* **Violence detected** → Red banner
* **Safe scene** → Green banner

---

##  Tech Stack

* **Python 3.8+**
* **PyTorch**
* **PyTorch Geometric**
* **Ultralytics YOLOv8**
* **MediaPipe**
* **Streamlit**
* **OpenCV**
* **Pillow (for Unicode rendering)**

---

##  Future Improvements

* Real‑time deployment on CCTV hardware
* Multi‑person violence interaction detection
* Audio‑visual fusion
* Transformer‑based GNN alternatives

---

## 💬 Contact

For questions or collaboration:
**Durbadal Bhowmik** — *[durbadal.bhowmik8617@gmail.com]*

---

## Support

If you like this project, please consider **starring the repository** on GitHub!
