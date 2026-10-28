"""Synthetic media generator for testing and demonstration in DeepShield AI."""

from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = BASE_DIR / "data" / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


def draw_synthetic_face(width: int = 400, height: int = 400) -> np.ndarray:
    """Render a synthetic facial pattern with geometry triggering face detection."""
    img = np.full((height, width, 3), 240, dtype=np.uint8)

    # Face contour / skin tone
    center = (width // 2, height // 2)
    axes = (width // 4, height // 3)
    cv2.ellipse(img, center, axes, 0, 0, 360, (180, 205, 235), -1)
    cv2.ellipse(img, center, axes, 0, 0, 360, (120, 140, 170), 2)

    # Eyes
    left_eye = (width // 2 - 45, height // 2 - 35)
    right_eye = (width // 2 + 45, height // 2 - 35)
    cv2.circle(img, left_eye, 16, (255, 255, 255), -1)
    cv2.circle(img, right_eye, 16, (255, 255, 255), -1)
    cv2.circle(img, left_eye, 8, (60, 40, 30), -1)
    cv2.circle(img, right_eye, 8, (60, 40, 30), -1)

    # Eyebrows
    cv2.line(img, (left_eye[0] - 20, left_eye[1] - 18), (left_eye[0] + 20, left_eye[1] - 16), (40, 30, 20), 3)
    cv2.line(img, (right_eye[0] - 20, right_eye[1] - 16), (right_eye[0] + 20, right_eye[1] - 18), (40, 30, 20), 3)

    # Nose
    nose_pt1 = (width // 2, height // 2 - 10)
    nose_pt2 = (width // 2 - 10, height // 2 + 20)
    nose_pt3 = (width // 2 + 10, height // 2 + 20)
    cv2.polylines(img, [np.array([nose_pt1, nose_pt2, nose_pt3])], isClosed=False, color=(140, 120, 100), thickness=2)

    # Mouth
    mouth_center = (width // 2, height // 2 + 55)
    cv2.ellipse(img, mouth_center, (35, 12), 0, 0, 180, (60, 60, 180), -1)

    return img


def generate_all_samples():
    """Generate synthetic test assets."""
    print("Generating synthetic media samples...")

    # 1. Synthetic Face Image
    face_img = draw_synthetic_face()
    face_path = SAMPLES_DIR / "sample_synthetic_face.jpg"
    cv2.imwrite(str(face_path), face_img)
    print(f"Created: {face_path}")

    # 2. No Face Image (pure gradient / landscape)
    no_face = np.zeros((300, 300, 3), dtype=np.uint8)
    for y in range(300):
        no_face[y, :, 0] = int(255 * (y / 300))
        no_face[y, :, 1] = int(120 * (1 - y / 300))
        no_face[y, :, 2] = 200
    no_face_path = SAMPLES_DIR / "sample_no_face.jpg"
    cv2.imwrite(str(no_face_path), no_face)
    print(f"Created: {no_face_path}")

    # 3. Synthetic Test Video (20 frames, 10 fps)
    vid_path = SAMPLES_DIR / "sample_test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(vid_path), fourcc, 10.0, (400, 400))
    for i in range(20):
        # Slightly jitter the synthetic face position to simulate video motion
        offset = int(5 * np.sin(i / 2.0))
        frame = draw_synthetic_face(400, 400)
        cv2.putText(frame, f"Frame {i+1}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        out.write(frame)
    out.release()
    print(f"Created: {vid_path}")


if __name__ == "__main__":
    generate_all_samples()
