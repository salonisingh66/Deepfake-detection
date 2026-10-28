"""Setup sample and evaluation assets for DeepShield AI.

Generates real public-domain samples from scikit-image and synthetic manipulated
samples for evaluation pipeline testing.
"""

from pathlib import Path
import cv2
import numpy as np
from skimage import data

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "data" / "samples"
EVAL_DIR = BASE_DIR / "data" / "evaluation"
REAL_EVAL_DIR = EVAL_DIR / "real"
FAKE_EVAL_DIR = EVAL_DIR / "fake"

for d in (SAMPLES_DIR, REAL_EVAL_DIR, FAKE_EVAL_DIR):
    d.mkdir(parents=True, exist_ok=True)


def setup_all_sample_assets():
    print("Preparing demonstration and evaluation media assets...")

    # 1. Real Face Photo (NASA astronaut photo - public domain)
    astronaut = data.astronaut()
    astronaut_bgr = cv2.cvtColor(astronaut, cv2.COLOR_RGB2BGR)
    real_sample_path = SAMPLES_DIR / "sample_face_real.jpg"
    cv2.imwrite(str(real_sample_path), astronaut_bgr)
    print(f"  [OK] Saved {real_sample_path}")

    # 2. No Face Photo (Moon / Rocket)
    rocket = data.rocket()
    rocket_bgr = cv2.cvtColor(rocket, cv2.COLOR_RGB2BGR)
    no_face_path = SAMPLES_DIR / "sample_no_face.jpg"
    cv2.imwrite(str(no_face_path), rocket_bgr)
    print(f"  [OK] Saved {no_face_path}")

    # 3. Video With Face (25 frames, 10 fps)
    h, w = astronaut.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vid_face_path = SAMPLES_DIR / "sample_video_with_face.mp4"
    out = cv2.VideoWriter(str(vid_face_path), fourcc, 10.0, (w, h))
    for i in range(25):
        # Subtle realistic lighting variation
        factor = 1.0 + 0.04 * np.sin(i / 2.5)
        frame = cv2.convertScaleAbs(astronaut_bgr, alpha=factor, beta=0)
        out.write(frame)
    out.release()
    print(f"  [OK] Saved {vid_face_path}")

    # 4. Video Without Face (Moon / Rocket video)
    rh, rw = rocket.shape[:2]
    vid_no_face_path = SAMPLES_DIR / "sample_video_no_face.mp4"
    out_no_face = cv2.VideoWriter(str(vid_no_face_path), fourcc, 10.0, (rw, rh))
    for i in range(20):
        out_no_face.write(rocket_bgr)
    out_no_face.release()
    print(f"  [OK] Saved {vid_no_face_path}")

    # 5. Populate Sample Evaluation Dataset (real vs manipulated)
    # Real samples: variations of real photograph (cropping, lighting, slight rotations)
    for idx, angle in enumerate([0, 2, -2, 4]):
        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
        rotated = cv2.warpAffine(astronaut_bgr, M, (w, h))
        cv2.imwrite(str(REAL_EVAL_DIR / f"real_sample_{idx+1}.jpg"), rotated)

    # Fake samples: simulated deepfake artifacts (facial warping, color mismatch, Gaussian smoothing on face)
    face_rect = (160, 50, 150, 170)  # Approximate face region in astronaut image
    fx, fy, fw, fh = face_rect
    for idx in range(4):
        fake_img = astronaut_bgr.copy()
        face_roi = fake_img[fy : fy + fh, fx : fx + fw]
        if idx == 0:
            # Heavy smoothing / skin beautification artifact
            blurred_roi = cv2.bilateralFilter(face_roi, 15, 80, 80)
            fake_img[fy : fy + fh, fx : fx + fw] = blurred_roi
        elif idx == 1:
            # Boundary seam / color tint mismatch
            tinted_roi = cv2.convertScaleAbs(face_roi, alpha=1.15, beta=15)
            fake_img[fy : fy + fh, fx : fx + fw] = tinted_roi
        elif idx == 2:
            # Compression / sharpening artifact on face
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharp_roi = cv2.filter2D(face_roi, -1, kernel)
            fake_img[fy : fy + fh, fx : fx + fw] = sharp_roi
        else:
            # Synthetic noise injection
            noise = np.random.normal(0, 15, face_roi.shape).astype(np.int16)
            noisy = np.clip(face_roi.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            fake_img[fy : fy + fh, fx : fx + fw] = noisy

        cv2.imwrite(str(FAKE_EVAL_DIR / f"fake_sample_{idx+1}.jpg"), fake_img)

    print("  [OK] Initialized evaluation dataset in data/evaluation/ (4 Real, 4 Fake samples)")
    print("Done.\n")


if __name__ == "__main__":
    setup_all_sample_assets()
