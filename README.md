# Multimodal Emotion Detection System

This project is a real-time multimodal emotion detection system that uses both facial expressions and speech to predict a person's emotion.

## How it Works

The system captures video from a webcam and audio from a microphone.
- A Convolutional Neural Network (CNN) analyzes the facial expression to predict the emotion.
- A machine learning model (like SVM or Logistic Regression) analyzes the audio features (MFCCs) to predict the emotion from speech.
- The predictions from both models are combined (fused) to produce a final, more accurate emotion prediction.

## Installation

1.  Clone the repository or download the source code.
2.  Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```
3.  Download the datasets:
    - **Face Model:** Download the facial emotion dataset from [Kaggle](https://www.kaggle.com/datasets/abhisheksingh016/machine-model-for-emotion-detection). Unzip it and place the `train` and `test` folders inside the `face_model` directory.
    - **Audio Model:** Download the RAVDESS dataset from [Zenodo](https://zenodo.org/records/1188976). Unzip it and place the `Audio_Speech_Actors_01-24` folder inside the `audio_model` directory.

## Training the Models

Before running the main application, you need to train the face and audio models.

1.  **Train the Face Model:**
    Navigate to the `face_model` directory and run the training script:
    ```bash
    cd face_model
    python train_face.py
    cd ..
    ```
    This will create a `model.h5` file in the `face_model` directory.

2.  **Train the Audio Model:**
    Navigate to the `audio_model` directory and run the training script:
    ```bash
    cd audio_model
    python train_audio.py
    cd ..
    ```
    This will create a `model.pkl` file in the `audio_model` directory.

## Running the Application

Once the models are trained and located in their respective directories, you can run the main application:

```bash
python main.py
```

- A window will open showing your webcam feed.
- The detected face, audio, and final emotions will be printed to the console and displayed on the screen.
- Press 'q' to quit the application.

## Multimodal Learning

Multimodal learning involves using data from multiple modalities (like image, audio, text) to make a prediction. This approach often leads to better and more robust models because some modalities can provide information that others miss. In this project, we use a simple late fusion technique where we average the probabilities from the face and audio models to get a combined prediction.
