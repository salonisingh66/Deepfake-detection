# DeepShield AI: Model Card

## Model Overview
- **Model Name:** ViT Deepfake Forensic Classifier
- **Model Identifier:** `vit_deepfake`
- **Backbone Architecture:** Vision Transformer (`vit-base-patch16-224`)
- **Developer / Fine-Tuner:** Dmitry Retinskiy (`dima806`)
- **Model Checkpoint Source:** [HuggingFace Hub: dima806/deepfake_vs_real_image_detection](https://huggingface.co/dima806/deepfake_vs_real_image_detection)
- **License:** Apache-2.0 / Open Access
- **Intended Use:** Digital forensic screening of facial crops for signs of manipulation, face-swapping, and GAN/diffusion facial generation.

## Input Specification
- **Input Type:** Cropped RGB facial images
- **Input Dimensions:** 224 x 224 pixels
- **Normalization:** Mean = (0.5, 0.5, 0.5), Standard Deviation = (0.5, 0.5, 0.5)
- **Batch Size:** 1 (single crop) or bounded chunks of 8 (CPU optimized)

## Output Specification
- **Task:** Binary Classification
- **Class 0:** Real (Authentic facial characteristics)
- **Class 1:** Fake (Manipulated or synthetic facial characteristics)
- **Softmax Normalization:** Output scores normalized to range `[0.0, 1.0]` representing class probabilities.

## Training & Evaluation Data
- **Training Datasets:** 140,000 real and manipulated faces derived from FaceForensics++, Celeb-DF v2, and the 140k Real/Fake Face dataset.
- **Manipulation Techniques Represented:** DeepFakes (encoder-decoder face swaps), Face2Face (expression re-enactment), FaceSwap (graphics blending), and NeuralTextures.

## Performance Characteristics (Laptop CPU - Intel/Apple Silicon)
- **Model Load Time:** ~1.4 seconds from local disk cache
- **Single Image Inference Latency:** ~700 - 1400 ms on CPU
- **Memory Footprint:** ~350 MB RAM for model weights

## Ethical Considerations & Limitations
- Prediction scores represent mathematical pattern sensitivity, not legal proof of guilt or innocence.
- Performance degrades on heavily compressed low-bitrate video frames (MPEG block artifacts).
- False positives may occur on extreme head poses (>45 degrees yaw/pitch) or low-light noisy captures.
- Always review results in conjunction with human expert verification and cryptographic provenance.
