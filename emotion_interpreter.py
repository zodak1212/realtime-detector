"""
emotion_interpreter.py

Maps a 7-class probability vector from the face emotion model into a
Plutchik-style compound emotion label, with temporal smoothing and optional
per-class calibration to compensate for real-time over/under-firing.

Reference:
    Plutchik, R. (2001). The Nature of Emotions.
    American Scientist, 89(4), 344-350.
"""
import numpy as np


# --- Plutchik dyads ---------------------------------------------------------
# Curated subset chosen for webcam-performability and to avoid relying on
# unreliable classes (over-firing disgust, under-trained fear). The full
# original set is preserved as commented entries below — uncomment any
# you want re-enabled.
PLUTCHIK_DYADS = {
    frozenset(('fear', 'surprise')):  'Awe',          # primary
    frozenset(('surprise', 'sad')):   'Disapproval',  # primary
    frozenset(('disgust', 'angry')):  'Contempt',     # primary
    frozenset(('happy', 'sad')):      'Bittersweet',  # tertiary
    frozenset(('angry', 'happy')):    'Pride',        # secondary
    # --- cut for now (too many to reliably perform / over-firing classes) ---
    # frozenset(('sad', 'disgust')):      'Remorse',      # disgust over-fires
    # frozenset(('happy', 'fear')):       'Guilt',        # fear under-trained
    # frozenset(('fear', 'sad')):         'Despair',      # fear under-trained
    # frozenset(('surprise', 'disgust')): 'Unbelief',     # disgust over-fires
    # frozenset(('sad', 'angry')):        'Envy',         # too close to Contempt
    # frozenset(('happy', 'disgust')):    'Morbidness',   # disgust over-fires
    # frozenset(('fear', 'angry')):       'Outrage',      # fear under-trained
}

# Plutchik's intensity gradations for a single dominant emotion.
INTENSITY_LADDER = {
    'happy':    ('Serenity',     'Joy',       'Ecstasy'),
    'sad':      ('Pensive',      'Sad',       'Grief'),
    'angry':    ('Annoyed',      'Angry',     'Rage'),
    'fear':     ('Apprehensive', 'Fearful',   'Terror'),
    'surprise': ('Distracted',   'Surprised', 'Amazement'),
    'disgust':  ('Bored',        'Disgust',   'Loathing'),
    'neutral':  ('Neutral',      'Neutral',   'Neutral'),
}


class CompoundEmotionInterpreter:
    """
    Convert successive 7-class probability vectors into a compound emotion
    label (Plutchik dyad) or, when no dyad is appropriate, an intensity-graded
    single-emotion label.
    """

    def __init__(self, labels,
                 ema_alpha=0.3,
                 dyad_min=0.18,
                 dyad_ratio=0.55,
                 intensity_thresholds=(0.40, 0.70),
                 class_weights=None):
        self.labels = list(labels)
        self.ema_alpha = float(ema_alpha)
        self.dyad_min = float(dyad_min)
        self.dyad_ratio = float(dyad_ratio)
        self.t_mid, self.t_high = intensity_thresholds
        self._smoothed = None

        if class_weights is None:
            self._weights = np.ones(len(self.labels), dtype=np.float32)
        else:
            self._weights = np.array(
                [float(class_weights.get(l, 1.0)) for l in self.labels],
                dtype=np.float32,
            )

    # ---- public API --------------------------------------------------------
    def reset(self):
        self._smoothed = None

    def update(self, probs):
        """Push a new probability vector. Returns the smoothed vector."""
        probs = np.asarray(probs, dtype=np.float32)

        if not np.allclose(self._weights, 1.0):
            probs = probs * self._weights
            probs = probs / max(float(probs.sum()), 1e-8)

        if self._smoothed is None:
            self._smoothed = probs.copy()
        else:
            a = self.ema_alpha
            self._smoothed = a * probs + (1.0 - a) * self._smoothed
        return self._smoothed

    def interpret(self):
        if self._smoothed is None:
            return None

        order = np.argsort(self._smoothed)[::-1]
        top_label = self.labels[order[0]]
        top_prob = float(self._smoothed[order[0]])
        sec_label = self.labels[order[1]]
        sec_prob = float(self._smoothed[order[1]])

        is_dyad = False
        compound = None

        if (top_label != 'neutral' and sec_label != 'neutral'
                and sec_prob >= self.dyad_min
                and sec_prob >= self.dyad_ratio * top_prob):
            key = frozenset((top_label, sec_label))
            if key in PLUTCHIK_DYADS:
                compound = PLUTCHIK_DYADS[key]
                is_dyad = True

        if not is_dyad:
            if top_prob >= self.t_high:
                idx = 2
            elif top_prob >= self.t_mid:
                idx = 1
            else:
                idx = 0
            compound = INTENSITY_LADDER.get(top_label, (top_label,) * 3)[idx]

        return {
            'top': (top_label, top_prob),
            'second': (sec_label, sec_prob),
            'compound': compound,
            'is_dyad': is_dyad,
            'smoothed_probs': self._smoothed.copy(),
        }