"""
predict_face.py
===============
Real-time face emotion prediction with temporal smoothing and compound emotions.
"""

import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from collections import deque
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import EmotionNet, EMOTION_LABELS

_predictor = None

class _FacePredictor:
    def __init__(self, smooth_window=5):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.pth")
        
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

        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.prob_history = deque(maxlen=smooth_window)
        self.emotion_history = deque(maxlen=30)

    def predict(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(48, 48))
        if len(faces) == 0: 
            return None, None
        
        # Select largest face
        x, y, w, h = faces[np.argmax([fw*fh for (_,_,fw,fh) in faces])]
        x, y = max(0, x), max(0, y) # Safety bounds
        
        crop = frame[y:y+h, x:x+w]
        if crop.size == 0:
            return None, None
            
        tensor = self.transform(crop).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            probs = F.softmax(self.model(tensor), dim=1).cpu().numpy()[0]
        
        self.prob_history.append(probs)
        avg_probs = np.mean(list(self.prob_history), axis=0)
        
        dom_idx = np.argmax(avg_probs)
        self.emotion_history.append(EMOTION_LABELS[dom_idx])
        
        return avg_probs, (x, y, w, h)

    def interpret_emotion(self, probs):
        """Compound emotions logic adapted for RAF-DB"""
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
    face_probs, face_coords = _get_predictor().predict(frame)
    return face_probs, face_coords

def interpret_emotion(probs):
    return _get_predictor().interpret_emotion(probs)