# DeepShield AI: Forensic Limitations and Caveats

DeepShield AI provides automated technical screening for digital forensics. Automated deepfake detection is subject to technical constraints that must be understood by investigators, journalists, and forensic analysts.

## 1. Mathematical Confidence vs Legal Authenticity
- A high "Fake Score" (e.g. 0.92) indicates that pixel patterns closely resemble training distribution artifacts (blending seams, warp boundaries, eye reflection anomalies).
- It does **not** constitute legal proof that an image was created by a specific generative model or individual.
- Conversely, a low "Fake Score" does **not** prove authenticity; highly sophisticated, post-processed, or analog re-photographed deepfakes can evade detection.

## 2. Correlated Frame Caveats in Video Processing
- Sequential video frames are temporally correlated; they do not represent independent statistical trials.
- Averaging frame-level scores produces a summary of central tendency, **not** a calibrated joint Bayesian probability of the entire video.
- DeepShield AI clearly distinguishes frame-level predictions from video-level summaries.

## 3. Compression and Transmission Degradation
- Social media platforms (WhatsApp, X/Twitter, Instagram, TikTok) re-compress media heavily using aggressive lossy codecs (H.264, VP9, JPEG at low quality factors).
- Heavy compression can obliterate high-frequency forensic artifacts (causing false negatives) or introduce synthetic block edge artifacts that confuse the model (causing false positives).

## 4. Metadata Forensics Caveats
- Absence of EXIF metadata is **normal** for media downloaded from social platforms, which automatically strip EXIF for user privacy.
- Missing EXIF is **not** evidence of deepfake generation.
- Presence of valid EXIF is easily forged using tools like `exiftool` and does not prove authenticity.

## 5. Visual Explanations (Attention Maps & Grad-CAM)
- Highlighted heatmaps indicate regions of **model sensitivity** (where the model attended during classification).
- They represent neural feature importance, **not** a highlighted bounding box of manipulated pixels.

## 6. Required Legal Disclaimer
> "This report contains automated model predictions and technical observations. It does not establish the authenticity or provenance of the media. Predictions may be incorrect, and results should not be used as the sole basis for legal, financial, disciplinary, or reputational decisions."
