"""
main.py — Performance testing dashboard
"""
import cv2
import numpy as np
import time
from face_model.predict_face import predict_face_emotion, EMOTION_LABELS

EMOTION_COLOURS = {
    'surprise': (50, 190, 255),
    'fear':     (200, 100, 200),
    'disgust':  (50, 160, 50),
    'happy':    (50, 210, 230),
    'sad':      (210, 160, 50),
    'angry':    (60, 60, 230),
    'neutral':  (180, 180, 180),
}

def main():
    cap = cv2.VideoCapture(0)
    prev_frame_time = 0
    new_frame_time = 0

    print("Dashboard running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Calculate FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps_text = f"FPS: {int(fps)}"

        # Predict Emotion
        res = predict_face_emotion(frame)
        
        if res and res[0] is not None:
            probs, (x, y, w, h) = res
            label = EMOTION_LABELS[np.argmax(probs)]
            color = EMOTION_COLOURS.get(label, (255, 255, 255))
            
            # Draw Face Box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label.upper(), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Draw Probabilities Sidebar
            for i, (name, p) in enumerate(zip(EMOTION_LABELS, probs)):
                cv2.putText(frame, f"{name}: {p:.2f}", (10, 60 + i*25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display FPS
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow('RAF-DB Real-Time Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()