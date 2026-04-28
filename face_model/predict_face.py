"""
predict_face.py
===============
Real-time face emotion prediction with temporal smoothing.

Face detection uses the modern MediaPipe Tasks API (BlazeFace short-range).
The .tflite model file is downloaded automatically to face_model/ on first run.

Note: the new Tasks API only ships the short-range face detector publicly,
which is slightly more frontal-focused than the legacy full-range model.
The grace period in main.py covers the occasional drop on extreme head tilt.
"""

import os
import sys
import urllib.request
from collections import deque

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import EmotionNet, EMOTION_LABELS

_predictor = None

_FACE_MODEL_FILE = "blaze_face_short_range.tflite"
_FACE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
)


def _ensure_face_model(path):
    """Download the BlazeFace .tflite to `path` if it is not already present."""
    if os.path.exists(path):
        return
    print(f"[predict_face] Downloading face detection model to {path} ...")
    try:
        urllib.request.urlretrieve(_FACE_MODEL_URL, path)
        print("[predict_face] Download complete.")
    except Exception as e:
        raise RuntimeError(
            f"Failed to download face detection model from {_FACE_MODEL_URL}. "
            f"Place it at {path} manually. Error: {e}"
        )


class _FacePredictor:
    def __init__(self, smooth_window=5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        here = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(here, "model.pth")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Run train_face.py first.")

        self.model = EmotionNet()
        ckpt = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device).eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # MediaPipe Tasks face detector. Auto-download model on first run.
        face_model_path = os.path.join(here, _FACE_MODEL_FILE)
        _ensure_face_model(face_model_path)
        options = mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=face_model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            min_detection_confidence=0.5,
        )
        self._face_detector = mp_vision.FaceDetector.create_from_options(options)

        self.prob_history = deque(maxlen=smooth_window)
        self.emotion_history = deque(maxlen=30)

    # ---- detection ---------------------------------------------------------
    def _detect_largest_face(self, frame):
        H, W, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._face_detector.detect(mp_image)

        if not result.detections:
            return None

        # Pick the largest detection. In the Tasks API, bounding_box is in
        # pixel coordinates (origin_x, origin_y, width, height).
        det = max(result.detections,
                  key=lambda d: d.bounding_box.width * d.bounding_box.height)
        bb = det.bounding_box

        # Pad ~10% on each side. MediaPipe boxes are tight; the emotion model
        # was trained on RAF-DB crops with forehead, cheeks, and jaw included.
        pad_x = int(bb.width * 0.10)
        pad_y = int(bb.height * 0.10)
        x = max(0, bb.origin_x - pad_x)
        y = max(0, bb.origin_y - pad_y)
        w = bb.width + 2 * pad_x
        h = bb.height + 2 * pad_y
        w = min(w, W - x)
        h = min(h, H - y)

        if w <= 0 or h <= 0:
            return None
        return x, y, w, h

    # ---- prediction --------------------------------------------------------
    def predict(self, frame):
        bbox = self._detect_largest_face(frame)
        if bbox is None:
            return None, None
        x, y, w, h = bbox

        crop = frame[y:y + h, x:x + w]
        if crop.size == 0:
            return None, None

        tensor = self.transform(crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs = F.softmax(self.model(tensor), dim=1).cpu().numpy()[0]

        self.prob_history.append(probs)
        avg_probs = np.mean(list(self.prob_history), axis=0)

        dom_idx = int(np.argmax(avg_probs))
        self.emotion_history.append(EMOTION_LABELS[dom_idx])

        return avg_probs, (x, y, w, h)

    # ---- legacy threshold-based compound emotions --------------------------
    # Kept for backward compatibility; superseded by emotion_interpreter.py.
    def interpret_emotion(self, probs):
        p = {e: float(v) for e, v in zip(EMOTION_LABELS, probs)}

        if p['happy'] >= 0.85:
            return "Very Happy"
        if p['happy'] >= 0.45 and p['surprise'] >= 0.25:
            return "Excited"
        if p['sad'] >= 0.35 and p['angry'] >= 0.30:
            return "Frustrated"
        if p['fear'] >= 0.35 and p['neutral'] >= 0.30:
            return "Anxious"
        if p['disgust'] >= 0.35 and p['angry'] >= 0.25:
            return "Contemptuous"
        if p['happy'] >= 0.30 and p['sad'] >= 0.25:
            return "Bittersweet"

        if len(self.emotion_history) >= 10:
            recent = list(self.emotion_history)[-10:]
            if len(set(recent)) >= 4:
                return "Confused"
            if recent[-1] == 'happy' and 'surprise' in recent[-5:]:
                return "Delighted"
            if recent.count('neutral') >= 8:
                return "Bored"
            if recent.count('angry') >= 3 and recent.count('neutral') >= 3:
                return "Agitated"

        return None


def _get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = _FacePredictor()
    return _predictor


def predict_face_emotion(frame):
    return _get_predictor().predict(frame)


def interpret_emotion(probs):
    return _get_predictor().interpret_emotion(probs)