"""
main.py — Performance testing dashboard with Plutchik-style compound emotions.
"""
import cv2
import numpy as np
import time

from face_model.predict_face import predict_face_emotion, EMOTION_LABELS
from emotion_interpreter import CompoundEmotionInterpreter


EMOTION_COLOURS = {
    'surprise': (50, 190, 255),
    'fear':     (200, 100, 200),
    'disgust':  (50, 160, 50),
    'happy':    (50, 210, 230),
    'sad':      (210, 160, 50),
    'angry':    (60, 60, 230),
    'neutral':  (180, 180, 180),
}
DYAD_COLOUR = (255, 255, 255)


# --- Tunables ----------------------------------------------------------------
CLASS_WEIGHTS = {
    'disgust': 0.55,
    'angry':   0.85,
    'sad':     1.30,
    'fear':    1.50,
}
# predict_face.py already does a 5-frame moving-average smooth via its deque,
# so disable the EMA here to avoid double-smoothing (which was making disgust
# spikes feel sticky). If you want extra smoothing on top, drop this to ~0.5;
# for less smoothing overall, reduce smooth_window in predict_face.py instead.
EMA_ALPHA = 1.0
INTENSITY_THRESHOLDS = (0.40, 0.85)

# Face-loss handling: keep showing last reading for a grace period so brief
# detector dropouts (head tilt, eyes closing on a sad expression, etc.) don't
# wipe the dashboard. Reset the smoother only after a longer absence.
FACE_LOSS_GRACE_S = 0.6
FACE_RESET_S = 2.0


def draw_label(frame, text, org, scale=0.7, colour=(255, 255, 255), thickness=2):
    """Text with a 1-pixel dark drop-shadow for readability against any background."""
    x, y = org
    cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, colour, thickness, cv2.LINE_AA)


def draw_dashboard(frame, interp, bbox, dim=False):
    """Render the full overlay (face box, banner, label, sidebar) for a given interpretation."""
    x, y, w, h = bbox
    top_label, top_prob = interp['top']
    sec_label, sec_prob = interp['second']
    compound = interp['compound']
    is_dyad = interp['is_dyad']
    smoothed = interp['smoothed_probs']

    box_colour = EMOTION_COLOURS.get(top_label, (255, 255, 255))
    if dim:  # face currently lost — show dimmed colour to signal stale state
        box_colour = tuple(int(c * 0.5) for c in box_colour)

    cv2.rectangle(frame, (x, y), (x + w, y + h), box_colour, 2)

    banner = compound.upper()
    if is_dyad:
        banner += f"  ({top_label} + {sec_label})"
    draw_label(frame, banner, (x, max(y - 38, 22)),
               scale=0.85, colour=DYAD_COLOUR, thickness=2)

    draw_label(frame, f"{top_label}: {top_prob:.2f}",
               (x, max(y - 12, 44)), scale=0.6, colour=box_colour)

    for i, (name, p) in enumerate(zip(EMOTION_LABELS, smoothed)):
        bar_colour = EMOTION_COLOURS.get(name, (220, 220, 220))
        if dim:
            bar_colour = tuple(int(c * 0.5) for c in bar_colour)
        bar_w = int(p * 140)
        cv2.rectangle(frame, (140, 60 + i * 24),
                      (140 + bar_w, 60 + i * 24 + 16), bar_colour, -1)
        draw_label(frame, f"{name}: {p:.2f}",
                   (10, 73 + i * 24), scale=0.55,
                   colour=(230, 230, 230), thickness=1)


def main():
    cap = cv2.VideoCapture(0)
    interpreter = CompoundEmotionInterpreter(
        EMOTION_LABELS,
        ema_alpha=EMA_ALPHA,
        intensity_thresholds=INTENSITY_THRESHOLDS,
        class_weights=CLASS_WEIGHTS,
    )

    prev_t = time.time()
    last_face_t = 0.0
    last_state = None  # tuple: (interp dict, bbox)

    print("Dashboard running. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        fps = 1.0 / max(now - prev_t, 1e-6)
        prev_t = now

        res = predict_face_emotion(frame)
        face_present = bool(res and res[0] is not None)

        if face_present:
            probs, bbox = res
            interpreter.update(probs)
            interp = interpreter.interpret()
            last_state = (interp, bbox)
            last_face_t = now

        time_since_face = now - last_face_t

        # --- decide what to draw ------------------------------------------
        if last_state is not None and time_since_face < FACE_LOSS_GRACE_S:
            interp, bbox = last_state
            draw_dashboard(frame, interp, bbox, dim=not face_present)

        # Reset smoother on prolonged loss so we don't carry stale probs.
        if time_since_face > FACE_RESET_S:
            interpreter.reset()
            last_state = None

        # --- status indicators --------------------------------------------
        h_frame = frame.shape[0]
        if not face_present:
            if last_state is not None:
                draw_label(frame, "FACE LOST (showing last)",
                           (10, h_frame - 15), scale=0.55,
                           colour=(0, 165, 255), thickness=1)
            else:
                draw_label(frame, "NO FACE DETECTED",
                           (10, h_frame - 15), scale=0.55,
                           colour=(0, 0, 255), thickness=1)

        draw_label(frame, f"FPS: {int(fps)}", (10, 30),
                   scale=0.7, colour=(255, 200, 0))

        cv2.imshow('Real-Time Emotion Detection (Plutchik dyads)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()