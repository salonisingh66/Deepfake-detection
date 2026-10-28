# DeepShield AI: Deepfake Detection and Forensic Analysis System

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.37%2B-FF4B4B.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Tests Passing](https://img.shields.io/badge/Tests-34%20Passed-success.svg)](tests/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

---

## 1. Overview & Problem Statement

**DeepShield AI** is a professional, web-based deepfake detection and digital forensics analysis system engineered to identify signs of AI-generated, synthetic, and manipulated facial media in images and video streams. 

With the rapid proliferation of generative adversarial networks (GANs), diffusion models, and real-time facial re-enactment tools, synthetic media poses severe risks to information integrity, identity verification, legal proceedings, and public trust. 

DeepShield AI is purpose-built to run locally on a standard laptop (**12 GB RAM, CPU-only processing, no dedicated NVIDIA GPU required**) while prioritizing:
1. **Functional correctness** with real machine learning inference.
2. **CPU and memory efficiency** via bounded batching and streaming frame sampling.
3. **Forensic provenance** through cryptographic hashing (SHA-256, MD5) and EXIF inspection.
4. **Transparent visual explanations** (ViT Attention Maps & Grad-CAM).
5. **Auditable reporting** via downloadable, court-admissible-style PDF forensic reports.

---

## 2. Key Features

- **🛡️ Modern Cyber-Forensics Dashboard:** Real-time telemetry, model readiness verification, system health, and navigation across forensic tools.
- **🔍 Image Deepfake Detection:** Safe ingestion of JPG, PNG, and WEBP formats; multi-face localization with margin padding; real PyTorch inference; manipulation probability gauge.
- **🎬 Video Deepfake Inspection:** Streaming OpenCV video reading; configurable frame sampling interval and max frame bounds; memory cleanup; interactive Plotly timeline; sampled frame inspector.
- **📈 Documented Video Aggregation:** Statistical summary reporting mean, median, score standard deviation, fake frame ratio, and temporal correlation caveats.
- **👁️ Visual Explanations:** Vision Transformer (ViT) patch self-attention rollout maps highlighting spatial regions of model sensitivity.
- **📄 Forensic PDF Reports:** One-click generation of professional PDF reports with ReportLab, complete with file hashes, timestamps, metadata observations, frame tables, and ethical disclaimers.
- **📜 SQLite Analysis Audit Trail:** Local SQLite database storing analysis metadata, verdicts, and hashes without ever retaining raw media bytes.
- **📁 Batch Image Analysis:** Concurrent processing of up to 20 images with summary statistics and CSV export.
- **📊 Model Evaluation Suite:** Benchmark tool calculating Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix on labeled test sets.
- **⚡ FastAPI REST API:** High-performance RESTful API endpoints (`/health`, `/model-info`, `/predict/image`, `/predict/video`) with OpenAPI documentation.

---

## 3. Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **User Interface** | Streamlit, Plotly, HTML5/CSS3 (Cyber-Forensics Dark Theme) |
| **Machine Learning** | PyTorch, Torchvision, HuggingFace Transformers, TIMM |
| **Default Model** | Vision Transformer (`dima806/deepfake_vs_real_image_detection`) |
| **Computer Vision** | OpenCV (Haar Cascades, VideoCapture), Pillow, scikit-image |
| **Reporting & Export** | ReportLab (Forensic PDFs), Pandas (CSV Export) |
| **Evaluation** | Scikit-learn (ROC-AUC, Precision/Recall, Confusion Matrix) |
| **Database** | SQLite3 (Audit log metadata only) |
| **REST API** | FastAPI, Uvicorn, Pydantic |
| **Testing** | Pytest (34 automated unit and integration tests) |

---

## 4. Project Structure

```
DeepShield-AI/
|-- app.py                        # Main Streamlit web dashboard
|-- requirements.txt              # Production dependency specifications
|-- README.md                     # Comprehensive documentation
|-- .gitignore                    # Excludes weights, temp media, local DBs
|-- .env.example                  # Environment configuration template
|
|-- config/
|   |-- __init__.py
|   |-- settings.py               # Operational bounds, limits, and paths
|   |-- model_config.py           # Model metadata, class maps, thresholds
|
|-- src/
|   |-- __init__.py
|   |-- models/
|   |   |-- __init__.py
|   |   |-- model_loader.py       # Singleton model cache manager
|   |   |-- inference.py          # Bounded inference runner
|   |   |-- preprocessing.py      # Deterministic resizing & normalization
|   |   |-- model_metadata.py     # Registry query interface
|   |   |-- architectures.py      # EfficientNet-B0 & MesoNet definitions
|   |
|   |-- detection/
|   |   |-- __init__.py
|   |   |-- face_detector.py      # OpenCV Haar cascade face detector
|   |   |-- image_analyzer.py     # End-to-end image pipeline
|   |   |-- video_analyzer.py     # Streaming frame-by-frame video analyzer
|   |   |-- aggregation.py        # Statistical video score aggregation
|   |
|   |-- forensics/
|   |   |-- __init__.py
|   |   |-- hashing.py            # SHA-256 and MD5 cryptographic hashing
|   |   |-- metadata.py           # EXIF and video container inspection
|   |   |-- explanations.py       # ViT Attention & Grad-CAM overlays
|   |   |-- report_generator.py   # ReportLab forensic PDF builder
|   |
|   |-- database/
|   |   |-- __init__.py
|   |   |-- db.py                 # SQLite connection & schema initialization
|   |   |-- history.py            # Audit log CRUD operations
|   |
|   |-- evaluation/
|   |   |-- __init__.py
|   |   |-- metrics.py            # Accuracy, F1, ROC-AUC calculation
|   |   |-- evaluate.py           # Dataset benchmark executor
|   |
|   |-- utils/
|       |-- __init__.py
|       |-- validation.py         # File security, extension & dimension checks
|       |-- file_utils.py         # Safe temp files & path traversal defense
|       |-- logging_config.py     # Structured logging
|
|-- api/
|   |-- __init__.py
|   |-- main.py                   # FastAPI REST API service
|
|-- scripts/
|   |-- setup_model.py            # Model download, cache & benchmark script
|   |-- evaluate_dataset.py       # CLI dataset evaluation tool
|   |-- prepare_sample_data.py    # Generates sample media for demonstrations
|
|-- tests/
|   |-- test_validation.py        # Validation unit tests
|   |-- test_hashing.py           # Cryptographic hash tests
|   |-- test_database.py          # SQLite database tests
|   |-- test_aggregation.py       # Video aggregation strategy tests
|   |-- test_image_pipeline.py    # Image pipeline integration tests
|   |-- test_video_pipeline.py    # Video streaming integration tests
|   |-- test_reports.py           # Forensic PDF generation tests
|   |-- test_api.py               # FastAPI endpoint tests
|
|-- data/
|   |-- samples/                  # Sample test images and video clips
|   |-- evaluation/               # Benchmark real/ and fake/ datasets
|   |-- temp/                     # Safe temporary frame buffers
|
|-- models/
|   |-- README.md                 # Model documentation
|   |-- checkpoints/              # Local checkpoint storage
|
|-- reports/
|   |-- README.md                 # Report directory notes
|
|-- docs/
    |-- architecture.md           # System design & component interactions
    |-- model_card.md             # Model specification, training data & bias
    |-- limitations.md            # Forensic caveats & legal disclaimer
    |-- testing.md                # Test suite documentation & coverage
```

---

## 5. Installation Instructions

### Prerequisites
- Python 3.11 or Python 3.12
- Git
- Recommended: 12 GB RAM or higher

### Windows (PowerShell)
```powershell
# 1. Clone repository
git clone https://github.com/your-username/deepshield-ai.git
cd deepshield-ai

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux
```bash
# 1. Clone repository
git clone https://github.com/your-username/deepshield-ai.git
cd deepshield-ai

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 6. Model Acquisition & Setup

DeepShield AI uses a genuine pretrained deepfake detection model: **`dima806/deepfake_vs_real_image_detection`** (Vision Transformer ViT-Base fine-tuned on 140,000 real and manipulated faces from FaceForensics++ and Celeb-DF v2).

Run the automated setup and verification script:

```bash
python scripts/setup_model.py
```

This script:
1. Downloads and caches the model weights (~340 MB) from Hugging Face Hub to your local cache.
2. Verifies checkpoint integrity and configuration.
3. Runs a CPU benchmark inference test to verify latency (~700 - 1400 ms on laptop CPU).

To generate demonstration media assets (NASA astronaut sample photo, test video with face, sample video without face, and evaluation benchmark sets):

```bash
python scripts/prepare_sample_data.py
```

---

## 7. How to Run the Application

### Launch Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Launch FastAPI REST API Service (Optional)
```bash
uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
```
Interactive API documentation will be available at:
- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

---

## 8. Running Automated Tests

Run the full automated test suite with Pytest:

```bash
pytest -v
```

Expected result:
```
============================== 34 passed in 8.19s ==============================
```

All 34 tests execute within ~8.2 seconds on CPU without network downloads.

---

## 9. Model Evaluation & Benchmark CLI

Evaluate the detector on any labeled dataset organized into `real/` and `fake/` subfolders:

```bash
python scripts/evaluate_dataset.py --data-dir data/evaluation --threshold 0.50
```

Output includes:
- Total Samples (Real vs Fake distribution)
- Accuracy, Precision, Recall, F1-Score, and ROC-AUC
- Formatted Confusion Matrix
- Export of per-sample findings to CSV

---

## 10. CPU Performance Considerations

DeepShield AI is engineered for constrained hardware:
- **Streaming Video:** Uses `cv2.VideoCapture` generator-style reads; frames are released from RAM immediately after face cropping.
- **Bounded Sampling:** Default sampling is set to every 15th frame, capped at 30-60 frames max, ensuring video processing completes in seconds.
- **Inference Mode:** Executes within `torch.inference_mode()` with autograd tracking disabled.
- **Singleton Model Cache:** Model weights are loaded once in memory and preserved across browser sessions.

---

## 11. Ethical Considerations & Mandatory Disclaimer

> **LEGAL & FORENSIC DISCLAIMER:**
> "This report contains automated model predictions and technical observations. It does not establish the authenticity or provenance of the media. Predictions may be incorrect, and results should not be used as the sole basis for legal, financial, disciplinary, or reputational decisions."

---

## 12. Model & Dataset Licenses

- **ViT Deepfake Model Weights:** Apache-2.0 License via Dmitry Retinskiy (`dima806/deepfake_vs_real_image_detection`).
- **FaceForensics++ Dataset:** Academic research use.
- **NASA Media:** Public domain imagery (NASA Astronaut Eileen Collins).
- **DeepShield AI Source Code:** Apache-2.0 License.
