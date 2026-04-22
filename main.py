"""
main.py
=======
Real-time emotion detection from webcam.

Displays the detected emotion, confidence, and all probabilities on screen.
Supports temporal smoothing and compound emotion inference.

Usage:
  python main.py
  python main.py --model checkpoints/best_model.pth --smooth-window 5
"""

import cv2
import numpy as np
import argparse
import time

from predict import EmotionPredictor, EMOTION_LABELS


# Colours for different emotion categories
EMOTION_COLOURS = {
    'angry':    (0, 0, 255),      # red
    'disgust':  (0, 140, 0),      # dark green
    'fear':     (200, 0, 200),    # purple
    'happy':    (0, 220, 220),    # yellow
    'sad':      (220, 150, 0),    # blue
    'surprise': (0, 200, 255),    # orange
    'neutral':  (200, 200, 200),  # grey
}


def get_emotion_colour(emotion: str):
    """Get the display colour for an emotion, including compound emotions."""
    lower = emotion.lower()
    # Map compound emotions to their base colour
    mapping = {
        'very happy': 'happy', 'excited': 'surprise', 'frustrated': 'angry',
        'anxious': 'fear', 'contemptuous': 'disgust', 'bittersweet': 'sad',
        'confused': 'neutral', 'delighted': 'happy', 'bored': 'neutral',
        'agitated': 'angry',
    }
    key = mapping.get(lower, lower)
    return EMOTION_COLOURS.get(key, (200, 200, 200))


def draw_overlay(frame, emotion, confidence, probs, face_box, fps):
    """Draw the emotion overlay on the frame."""
    h, w = frame.shape[:2]
    colour = get_emotion_colour(emotion)

    # Face rectangle
    if face_box:
        x, y, fw, fh = face_box
        cv2.rectangle(frame, (x, y), (x + fw, y + fh), colour, 2)
        # Emotion label above face
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2)

    # Dark overlay panel
    panel_w = 260
    panel_h = 60 + len(EMOTION_LABELS) * 22
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + panel_w, 8 + panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Main emotion text
    display = f"{emotion} ({confidence * 100:.0f}%)"
    cv2.putText(frame, display, (18, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, colour, 2)

    # Probability bars
    y_offset = 55
    if probs:
        for label in EMOTION_LABELS:
            prob = probs.get(label, 0.0)
            bar_colour = EMOTION_COLOURS.get(label, (200, 200, 200))

            # Label
            cv2.putText(frame, f"{label:>8s}", (18, y_offset + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

            # Bar background
            bar_x = 95
            bar_w = 140
            cv2.rectangle(frame, (bar_x, y_offset - 8), (bar_x + bar_w, y_offset + 2), (60, 60, 60), -1)

            # Bar fill
            fill_w = int(bar_w * prob)
            if fill_w > 0:
                cv2.rectangle(frame, (bar_x, y_offset - 8), (bar_x + fill_w, y_offset + 2), bar_colour, -1)

            # Percentage
            cv2.putText(frame, f"{prob*100:.0f}%", (bar_x + bar_w + 5, y_offset + 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

            y_offset += 22

    # FPS
    cv2.putText(frame, f"FPS: {fps:.0f}", (w - 90, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def main(args):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    predictor = EmotionPredictor(
        model_path=args.model,
        smooth_window=args.smooth_window,
    )

    print("Starting emotion detection. Press 'q' to quit.")

    frame_count = 0
    fps_start = time.time()
    fps_display = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get full probabilities and smoothed prediction
        probs, face_box = predictor.get_all_probabilities(frame)
        emotion, confidence, face_box = predictor.predict_smoothed(frame)

        if emotion is None:
            emotion = "No Face"
            confidence = 0.0

        # FPS
        frame_count += 1
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_display = frame_count / elapsed
            frame_count = 0
            fps_start = time.time()

        # Draw
        draw_overlay(frame, emotion, confidence, probs, face_box, fps_display)
        cv2.imshow('Emotion Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth')
    parser.add_argument('--smooth-window', type=int, default=5)

    args = parser.parse_args()
    main(args)
