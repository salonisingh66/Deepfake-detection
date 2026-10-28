# Models Directory

This directory contains model configuration documentation and local checkpoints.

## Supported Architectures
1. **ViT Deepfake Forensic Classifier (`vit_deepfake`)**
   - Source: HuggingFace Hub (`dima806/deepfake_vs_real_image_detection`)
   - Backbone: Vision Transformer (ViT-Base-Patch16-224)
   - Status: Genuine pretrained model fine-tuned on 140k real and fake facial samples.
   - Cache Location: Automatically cached in `~/.cache/huggingface/hub/` for offline use.

2. **EfficientNet-B0 Deepfake Baseline (`efficientnet_b0`)**
   - Architecture: Compound scaling CNN adapted for binary classification.
   - Custom weights path: `models/checkpoints/efficientnet_b0_deepfake.pt`

3. **MesoInception-4 (`mesonet`)**
   - Architecture: Microscopic artifact forensic CNN (Afchar et al., 2018).
   - Custom weights path: `models/checkpoints/mesonet_deepfake.pt`

## Integrity & Safe Loading
- Checkpoint files (`*.bin`, `*.safetensors`, `*.pt`) are strictly excluded from Git tracking via `.gitignore`.
- Run `python scripts/setup_model.py` to verify weights and benchmark CPU latency.
