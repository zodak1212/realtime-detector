import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import joblib

# --- Constants and Configuration ---
# Path to the RAVDESS dataset
DATASET_PATH = os.path.join(os.path.dirname(__file__), 'Audio_Speech_Actors_01-24')

# Model output path
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')

# Emotion labels from RAVDESS dataset
# The mapping is based on the filename convention of RAVDESS
# 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
# We will use a subset for this project to match the face model.
# Let's map RAVDESS to our target emotions.
EMOTION_MAP = {
    '01': 'neutral', # neutral
    '03': 'happy',   # happy
    '04': 'sad',     # sad
    '05': 'angry',   # angry
    '06': 'fear',    # fear
    '08': 'surprise' # surprised
}

# The labels we want to predict
TARGET_LABELS = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def extract_features(file_path):
    """
    Extracts a rich set of acoustic features from an audio file.

    Features (194 total):
      - MFCC mean + std (40×2 = 80)
      - MFCC delta mean + std (40×2 = 80)
      - Chroma mean + std (12×2 = 24)
      - Spectral centroid, bandwidth, rolloff mean + std (3×2 = 6)
      - Zero crossing rate mean + std (2)
      - RMS energy mean + std (2)

    Args:
        file_path (str): Path to the audio file.

    Returns:
        numpy.ndarray: Feature vector, or None if an error occurs.
    """
    try:
        y, sr = librosa.load(file_path, duration=3, offset=0.5)
        features = []

        # MFCC mean and std
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        features.extend(np.mean(mfcc, axis=1))
        features.extend(np.std(mfcc, axis=1))

        # MFCC delta (velocity) mean and std — captures temporal dynamics
        mfcc_delta = librosa.feature.delta(mfcc)
        features.extend(np.mean(mfcc_delta, axis=1))
        features.extend(np.std(mfcc_delta, axis=1))

        # Chroma — pitch class energy (useful for distinguishing emotional tone)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))

        # Spectral centroid — "brightness" of sound
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(float(np.mean(spectral_centroid)))
        features.append(float(np.std(spectral_centroid)))

        # Spectral bandwidth — spread around the centroid
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features.append(float(np.mean(spectral_bandwidth)))
        features.append(float(np.std(spectral_bandwidth)))

        # Spectral rolloff — frequency below which 85% of energy lies
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.append(float(np.mean(spectral_rolloff)))
        features.append(float(np.std(spectral_rolloff)))

        # Zero crossing rate — noisiness / voicing indicator
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(float(np.mean(zcr)))
        features.append(float(np.std(zcr)))

        # RMS energy — overall loudness / arousal
        rms = librosa.feature.rms(y=y)
        features.append(float(np.mean(rms)))
        features.append(float(np.std(rms)))

        return np.array(features, dtype=np.float32)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def load_data():
    """
    Loads the RAVDESS dataset, extracts features, and returns them with labels.

    Returns:
        tuple: A tuple containing features (X) and labels (y).
    """
    features = []
    labels = []

    print("Loading dataset and extracting features...")
    # Iterate through each actor folder in the dataset
    for actor_folder in os.listdir(DATASET_PATH):
        actor_path = os.path.join(DATASET_PATH, actor_folder)
        if not os.path.isdir(actor_path):
            continue

        # Iterate through each audio file for the actor
        for file_name in os.listdir(actor_path):
            # The emotion is encoded in the filename, e.g., "03-01-03-01-01-01-01.wav"
            # The 3rd part "03" is the emotion.
            parts = file_name.split('-')
            if len(parts) > 2:
                emotion_code = parts[2]
                
                # Check if this emotion is one we want to use
                if emotion_code in EMOTION_MAP:
                    emotion_label = EMOTION_MAP[emotion_code]
                    
                    # Extract features from the audio file
                    file_path = os.path.join(actor_path, file_name)
                    mfcc_features = extract_features(file_path)
                    
                    if mfcc_features is not None:
                        features.append(mfcc_features)
                        labels.append(emotion_label)

    print(f"Loaded {len(features)} audio files.")
    return np.array(features), np.array(labels)

def train_model():
    """
    Loads data, trains an MLP model, and saves it.
    """
    # Load data
    X, y_str = load_data()

    if X.shape[0] == 0:
        print("No data was loaded. Aborting training.")
        return

    # --- Label Encoding ---
    # MLPClassifier's early_stopping validation uses np.isnan internally, which fails
    # on string labels. Encoding to integers avoids this sklearn limitation.
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)
    print("Label encoding:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # --- Feature Scaling ---
    # Standardise features to zero mean and unit variance before the MLP
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # --- Build and Train the Model ---
    # MLP with two hidden layers; early_stopping prevents overfitting on small datasets
    model = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation='relu',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        verbose=True,
    )

    print("Training the audio model...")
    model.fit(X_train, y_train)

    # --- Evaluate the Model ---
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")

    # --- Save the Trained Model, Scaler, and Classes ---
    # Store label_encoder.classes_ (the original string labels) so that predict_audio.py
    # can map model output indices back to emotion names without any changes.
    # model.classes_ would contain integers here, which is why we save the encoder's classes instead.
    model_data = {
        'model': model,
        'classes': label_encoder.classes_,
        'scaler': scaler,
    }
    joblib.dump(model_data, MODEL_PATH)
    print(f"Model and class labels saved to {MODEL_PATH}")
    print("Classes order:", label_encoder.classes_)

if __name__ == '__main__':
    # Check if dataset directory exists
    if not os.path.exists(DATASET_PATH):
        print("="*50)
        print("ERROR: Dataset not found.")
        print(f"Please download the RAVDESS dataset and place the 'Audio_Speech_Actors_01-24' folder in the '{os.path.dirname(__file__)}' directory.")
        print("="*50)
    else:
        train_model()
