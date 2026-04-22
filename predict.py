"""
predict.py
==========
Real-time face emotion prediction with temporal smoothing and compound emotion inference.

Usage:
  from predict import EmotionPredictor

  predictor = EmotionPredictor("checkpoints/best_model.pth")
  emotion, confidence, face_box = predictor.predict_smoothed(frame)
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from collections import deque
import time

from model import EmotionNet, EMOTION_LABELS


class EmotionPredictor:
    def __init__(self, model_path: str, device: str = None, smooth_window: int = 5):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Load model
        self.model = EmotionNet()
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        val_acc = checkpoint.get('val_acc', 'unknown')
        print(f"[Predictor] Model loaded (val_acc: {val_acc}) on {self.device}")

        # Preprocessing — must match training (no augmentation)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Face detector
        self._init_face_detector()

        # Temporal smoothing
        self.smooth_window = smooth_window
        self.prob_history = deque(maxlen=smooth_window)

        # Emotion history for compound inference
        self.emotion_history = deque(maxlen=30)  # ~1 second at 30fps
        self.emotion_timestamps = deque(maxlen=30)

    def _init_face_detector(self):
        """Use Haar cascade face detector."""
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError(f"Failed to load cascade from {cascade_path}")
        print("[Predictor] Face detector ready (Haar cascade)")

    def _detect_face(self, frame):
        """Detect the largest face in the frame."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48)
        )

        if len(faces) > 0:
            areas = [w * h for (_, _, w, h) in faces]
            idx = np.argmax(areas)
            x, y, w, h = faces[idx]
            face_crop = gray[y:y+h, x:x+w]
            if face_crop.size > 0:
                return face_crop, (x, y, w, h)

        return None, None

    def predict(self, frame):
        """Single-frame prediction. Returns (emotion, confidence, face_box) or (None, 0, None)."""
        face_crop, face_box = self._detect_face(frame)

        if face_crop is None:
            return None, 0.0, None

        input_tensor = self.transform(face_crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        idx = np.argmax(probs)
        return EMOTION_LABELS[idx], float(probs[idx]), face_box

    def predict_smoothed(self, frame):
        """
        Prediction with temporal smoothing + compound emotion inference.
        Returns (emotion, confidence, face_box).
        """
        face_crop, face_box = self._detect_face(frame)

        if face_crop is None:
            return None, 0.0, None

        input_tensor = self.transform(face_crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        # Temporal smoothing
        self.prob_history.append(probs)
        avg_probs = np.mean(list(self.prob_history), axis=0)

        # Base emotion from smoothed probabilities
        idx = np.argmax(avg_probs)
        base_emotion = EMOTION_LABELS[idx]
        confidence = float(avg_probs[idx])

        # Track for compound inference
        now = time.time()
        self.emotion_history.append(base_emotion)
        self.emotion_timestamps.append(now)

        # Try compound emotion inference
        compound = self._infer_compound_emotion(avg_probs, base_emotion, confidence)
        if compound:
            return compound, confidence, face_box

        return base_emotion, confidence, face_box

    def _infer_compound_emotion(self, probs, base_emotion, confidence):
        """
        Infer compound/complex emotions from probability patterns and temporal context.
        Returns a compound emotion string or None.
        """
        prob_dict = {e: float(p) for e, p in zip(EMOTION_LABELS, probs)}

        # --- Probability-based compounds ---

        # Very Happy: sustained high happiness
        if prob_dict['happy'] >= 0.85:
            return "Very Happy"

        # Excited: happy + surprise together
        if prob_dict['happy'] >= 0.45 and prob_dict['surprise'] >= 0.25:
            return "Excited"

        # Frustrated: sad + angry together
        if prob_dict['sad'] >= 0.35 and prob_dict['angry'] >= 0.30:
            return "Frustrated"

        # Anxious: fear + neutral (suppressed fear)
        if prob_dict['fear'] >= 0.35 and prob_dict['neutral'] >= 0.30:
            return "Anxious"

        # Contemptuous: disgust + angry
        if prob_dict['disgust'] >= 0.35 and prob_dict['angry'] >= 0.25:
            return "Contemptuous"

        # Bittersweet: happy + sad
        if prob_dict['happy'] >= 0.30 and prob_dict['sad'] >= 0.25:
            return "Bittersweet"

        # --- Temporal-based compounds (need history) ---
        if len(self.emotion_history) >= 10:
            recent = list(self.emotion_history)[-10:]
            timestamps = list(self.emotion_timestamps)[-10:]
            time_span = timestamps[-1] - timestamps[0]

            if time_span > 0:
                # Confused: rapid alternation between emotions, no dominant one
                unique_recent = set(recent)
                if len(unique_recent) >= 4:
                    return "Confused"

                # Delighted: surprise followed by happy
                if recent[-1] == 'happy' and 'surprise' in recent[-5:]:
                    return "Delighted"

                # Bored: prolonged neutral
                neutral_count = recent.count('neutral')
                if neutral_count >= 8:
                    return "Bored"

                # Agitated: rapid alternation between angry and neutral
                angry_count = recent.count('angry')
                neutral_count = recent.count('neutral')
                if angry_count >= 3 and neutral_count >= 3:
                    return "Agitated"

        return None

    def get_all_probabilities(self, frame):
        """Get full probability distribution. Returns (prob_dict, face_box) or (None, None)."""
        face_crop, face_box = self._detect_face(frame)

        if face_crop is None:
            return None, None

        input_tensor = self.transform(face_crop).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        prob_dict = {label: float(prob) for label, prob in zip(EMOTION_LABELS, probs)}
        return prob_dict, face_box
