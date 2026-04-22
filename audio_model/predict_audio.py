import sounddevice as sd
import numpy as np
import librosa
import joblib
import os

# --- Constants and Configuration ---
# Model path
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

# Audio recording settings
SAMPLE_RATE = 22050  # Must match the sample rate used during training
DURATION = 3  # seconds to record — matches the 3s window used during training
CHANNELS = 1 # Mono audio

# The order of emotion labels the final system will use
# This must be consistent across all parts of the project
FINAL_EMOTION_LABELS = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# --- Initialization ---
try:
    # Load the model, scaler, and class labels
    model_data = joblib.load(MODEL_PATH)
    audio_model = model_data['model']
    model_classes = model_data['classes']
    audio_scaler = model_data.get('scaler', None)
    print("Audio model loaded successfully.")
    print("Model was trained with these classes:", model_classes)
except Exception as e:
    print(f"Error loading audio model: {e}")
    print(f"Please ensure the model file '{MODEL_PATH}' exists and was trained correctly.")
    audio_model = None
    model_classes = []
    audio_scaler = None


def _extract_features_from_audio(y, sr):
    """
    Extracts the same rich feature set used during training.
    Must stay in sync with extract_features() in train_audio.py.
    """
    features = []

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    mfcc_delta = librosa.feature.delta(mfcc)
    features.extend(np.mean(mfcc_delta, axis=1))
    features.extend(np.std(mfcc_delta, axis=1))

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    features.extend(np.mean(chroma, axis=1))
    features.extend(np.std(chroma, axis=1))

    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(float(np.mean(spectral_centroid)))
    features.append(float(np.std(spectral_centroid)))

    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(float(np.mean(spectral_bandwidth)))
    features.append(float(np.std(spectral_bandwidth)))

    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(float(np.mean(spectral_rolloff)))
    features.append(float(np.std(spectral_rolloff)))

    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(float(np.mean(zcr)))
    features.append(float(np.std(zcr)))

    rms = librosa.feature.rms(y=y)
    features.append(float(np.mean(rms)))
    features.append(float(np.std(rms)))

    return np.array(features, dtype=np.float32)

def predict_audio_emotion():
    """
    Records a short audio clip, extracts features, and predicts the emotion.

    Returns:
        numpy.ndarray: A probability vector for the detected emotions, ordered according to FINAL_EMOTION_LABELS.
                       Returns None if the model is not loaded or an error occurs.
    """
    if audio_model is None:
        print("Audio model not loaded, cannot predict.")
        return None

    try:
        # --- Record Audio ---
        recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, dtype='float32')
        sd.wait()  # Wait until recording is finished

        # The recording is a numpy array. Librosa expects a 1D array, so we flatten it.
        y = recording.flatten()

        # --- Feature Extraction ---
        features = _extract_features_from_audio(y, SAMPLE_RATE).reshape(1, -1)

        # Apply the same scaler used during training
        if audio_scaler is not None:
            features = audio_scaler.transform(features)

        # --- Prediction ---
        probabilities = audio_model.predict_proba(features)[0]
        
        # --- Align Probabilities with Final Labels ---
        # Create a zero vector for the final probabilities
        final_probs = np.zeros(len(FINAL_EMOTION_LABELS))
        
        # Map the model's output probabilities to the correct positions in the final_probs array
        for i, emotion in enumerate(model_classes):
            if emotion in FINAL_EMOTION_LABELS:
                j = FINAL_EMOTION_LABELS.index(emotion)
                final_probs[j] = probabilities[i]
                
        return final_probs

    except Exception as e:
        print(f"An error occurred during audio prediction: {e}")
        return None

if __name__ == '__main__':
    # This block is for testing the prediction function directly
    if audio_model is None:
        print("Exiting test: Audio model is not available.")
        exit()

    print("\nStarting audio emotion prediction test...")
    print(f"Final labels order: {FINAL_EMOTION_LABELS}")
    
    # Perform a prediction
    predicted_probs = predict_audio_emotion()
    
    if predicted_probs is not None:
        print("\n--- Prediction Results ---")
        for emotion, prob in zip(FINAL_EMOTION_LABELS, predicted_probs):
            print(f"{emotion}: {prob * 100:.2f}%")
            
        # Get the top emotion
        top_emotion_index = np.argmax(predicted_probs)
        top_emotion = FINAL_EMOTION_LABELS[top_emotion_index]
        
        top_emotion_prob = predicted_probs[top_emotion_index]
        print(f"\nPredicted Emotion: {top_emotion} ({top_emotion_prob * 100:.2f}%)")
    else:
        print("Could not get a prediction.")
