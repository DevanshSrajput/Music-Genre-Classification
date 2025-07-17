"""Configuration settings for the Music Genre Classification project."""

import os

# Dataset configuration
DATASET_PATH = "data"
GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 
          'jazz', 'metal', 'pop', 'reggae', 'rock']

# Audio processing parameters
SAMPLE_RATE = 22050
DURATION = 30  # seconds
N_MFCC = 13
N_CHROMA = 12
N_SPECTRAL_CONTRAST = 7

# Model parameters
MODEL_PATH = "models"
TRAINED_MODEL_PATH = os.path.join(MODEL_PATH, "genre_classifier.h5")
SCALER_PATH = os.path.join(MODEL_PATH, "feature_scaler.pkl")

# Training parameters
BATCH_SIZE = 32
EPOCHS = 100
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1
RANDOM_STATE = 42

# Streamlit configuration
MAX_FILE_SIZE = 10  # MB
SUPPORTED_FORMATS = ['wav', 'mp3', 'flac', 'ogg']

# Create directories
os.makedirs(MODEL_PATH, exist_ok=True)
os.makedirs(DATASET_PATH, exist_ok=True)
os.makedirs("results", exist_ok=True)