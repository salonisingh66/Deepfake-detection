# DeepShield AI: System Architecture

## 1. Architectural Overview

DeepShield AI is organized as a modular, layered forensic system designed for reliable execution on standard laptops (12 GB RAM, CPU-only):

```
+-------------------------------------------------------------+
|                      User Interfaces                        |
|   Streamlit Web Dashboard (app.py)  |  FastAPI REST API     |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|                     Application Layer                       |
|   ImageAnalyzer  |  VideoAnalyzer  |  DatasetEvaluator      |
+-------------------------------------------------------------+
         |                     |                    |
         v                     v                    v
+------------------+  +------------------+  +-----------------+
| Forensics Module |  | Detection Module |  | Inference Engine|
| - SHA256 / MD5   |  | - Face Detector  |  | - ModelLoader   |
| - EXIF Parser    |  |   (Haar Cascade) |  |   (Singleton)   |
| - Attention Map  |  | - Margin Padder  |  | - Preprocessing |
| - ReportLab PDF  |  | - Score Aggreg.  |  | - Batch Runner  |
+------------------+  +------------------+  +-----------------+
                               |
                               v
+-------------------------------------------------------------+
|                    Persistence & Storage                    |
|       SQLite (Metadata Only - No biometric/media storage)   |
|       models/ (Pretrained weights & safe safetensors)       |
+-------------------------------------------------------------+
```

## 2. Core Subsystems

### 2.1 File Validation & Integrity Layer (`src/utils/validation.py`, `src/forensics/hashing.py`)
- Enforces strict file extension whitelisting (`.jpg`, `.jpeg`, `.png`, `.webp`, `.mp4`, `.mov`, `.avi`).
- Memory-bounded streaming calculation of SHA-256 and MD5 fingerprints.
- Verifies image integrity via PIL and video container properties via OpenCV before downstream memory allocation.

### 2.2 Facial Detection & Normalization (`src/detection/face_detector.py`)
- Employs OpenCV Haar Cascades for reliable, zero-dependency CPU face detection across Python 3.12 and macOS.
- Incorporates margin expansion (20% padding) around bounding boxes to capture forehead hair boundaries and jawline blending seams.
- Filters out small/noisy regions (< 40px) to prevent false positives.

### 2.3 Model Inference Engine (`src/models/`)
- Singleton `ModelLoader` loads model weights once into CPU memory.
- Uses `torch.inference_mode()` to disable autograd tracking and minimize memory consumption.
- Applies deterministic transforms matching the pretraining specification.
- Primary production detector: ViT (Vision Transformer) fine-tuned on 140k real and fake facial samples (`dima806/deepfake_vs_real_image_detection`).
- Baseline backbones: EfficientNet-B0 and MesoInception-4.

### 2.4 Video Frame-by-Frame Streaming & Aggregation (`src/detection/video_analyzer.py`, `src/detection/aggregation.py`)
- Streams video frames sequentially without loading entire videos into RAM.
- Re-encodes small thumbnails for timeline previews while immediately freeing full resolution frames.
- Guaranteed release of OpenCV resources in `finally:` blocks.
- Aggregates frame scores into median, mean, standard deviation, and fake frame ratios with documented statistical caveats.

### 2.5 Audit Database (`src/database/`)
- Persistent SQLite database storing analysis records.
- Privacy by design: Never stores images, video bytes, or biometric embeddings.
