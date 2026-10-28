# Data Directory

This directory stores demonstration media, evaluation datasets, and temporary processing buffers.

## Subdirectories
- `samples/`: Sample images and video clips for demonstration and pipeline verification (e.g., `sample_face_real.jpg`, `sample_no_face.jpg`, `sample_video_with_face.mp4`).
- `evaluation/`: Benchmark dataset organized into `real/` and `fake/` subdirectories for model evaluation metrics.
- `temp/`: Safe temporary folder for video chunking and frame processing (automatically cleaned up).
- `deepshield_history.db`: SQLite database storing forensic metadata audit logs (no media bytes stored).
