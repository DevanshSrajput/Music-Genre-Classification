"""Data loading and preprocessing for training."""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from tqdm import tqdm
from utils.audio_processor import AudioProcessor
from config import *

class DataLoader:
    def __init__(self, use_precomputed_features=True):
        self.audio_processor = AudioProcessor()
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.use_precomputed_features = use_precomputed_features
        
    def load_precomputed_features(self, csv_path):
        """Load pre-extracted features from CSV file."""
        print(f"Loading pre-computed features from {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"Dataset shape: {df.shape}")
        
        # Find label column
        if 'label' in df.columns:
            label_col = 'label'
        elif 'genre' in df.columns:
            label_col = 'genre'
        else:
            label_col = df.columns[-1]
        
        print(f"Using '{label_col}' as label column")
        
        # Get feature columns
        feature_cols = [col for col in df.columns if col not in [label_col, 'filename', 'track_id']]
        
        features = df[feature_cols].values
        labels = df[label_col].values
        
        print(f"Features shape: {features.shape}")
        print(f"Unique labels: {np.unique(np.array(labels))}")
        
        return features, labels, feature_cols
    
    def load_from_audio_files(self, dataset_path):
        """Load dataset from audio files."""
        print("Loading dataset from audio files...")
        features = []
        labels = []
        
        genre_folders = [d for d in os.listdir(dataset_path) 
                        if os.path.isdir(os.path.join(dataset_path, d))]
        
        for genre in tqdm(genre_folders, desc="Processing genres"):
            genre_path = os.path.join(dataset_path, genre)
            
            for filename in tqdm(os.listdir(genre_path), desc=f"Processing {genre}", leave=False):
                if filename.endswith('.wav'):
                    file_path = os.path.join(genre_path, filename)
                    try:
                        feature_vector, _ = self.audio_processor.process_file(file_path)
                        features.append(feature_vector)
                        labels.append(genre)
                    except Exception as e:
                        print(f"Error processing {file_path}: {str(e)}")
                        continue
        
        return np.array(features), np.array(labels)
    
    def load_dataset(self, dataset_path=DATASET_PATH, features_csv=None):
        """Load dataset using the best available method."""
        
        if self.use_precomputed_features and features_csv:
            features, labels, feature_names = self.load_precomputed_features(features_csv)
            self.feature_names = feature_names
            
        elif self.use_precomputed_features:
            # Try to find CSV files automatically
            possible_csv_files = [
                os.path.join(dataset_path, "features_30_sec.csv"),
                os.path.join(dataset_path, "features_3_sec.csv"),
                "features_30_sec.csv",
                "features_3_sec.csv"
            ]
            
            csv_file = None
            for csv_path in possible_csv_files:
                if os.path.exists(csv_path):
                    csv_file = csv_path
                    print(f"Found features CSV: {csv_file}")
                    break
            
            if csv_file:
                features, labels, feature_names = self.load_precomputed_features(csv_file)
                self.feature_names = feature_names
            else:
                print("No CSV files found, falling back to audio file processing...")
                genres_path = os.path.join(dataset_path, "genres_original")
                if os.path.exists(genres_path):
                    features, labels = self.load_from_audio_files(genres_path)
                    self.feature_names = self.get_feature_names()
                else:
                    raise FileNotFoundError(f"Neither CSV features nor audio files found in {dataset_path}")
        else:
            genres_path = os.path.join(dataset_path, "genres_original")
            if os.path.exists(genres_path):
                features, labels = self.load_from_audio_files(genres_path)
                self.feature_names = self.get_feature_names()
            else:
                raise FileNotFoundError(f"Audio files not found in {genres_path}")
        
        return features, labels
    
    def prepare_data(self, features, labels):
        """Prepare data for training."""
        # Clean labels
        cleaned_labels = []
        for label in labels:
            if isinstance(label, str):
                clean_label = label.lower().strip()
                clean_label = clean_label.replace('.wav', '').replace('.mp3', '')
                if '.' in clean_label:
                    clean_label = clean_label.split('.')[0]
                cleaned_labels.append(clean_label)
            else:
                cleaned_labels.append(str(label).lower().strip())
        
        # Encode labels
        labels_encoded = self.label_encoder.fit_transform(cleaned_labels)
        
        # Handle infinite/NaN values
        features_clean = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features_clean)
        
        print(f"Label mapping: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")
        
        return features_scaled, labels_encoded
    
    def split_data(self, features, labels, test_size=TEST_SPLIT, val_size=VALIDATION_SPLIT, random_state=RANDOM_STATE):
        """Split data into train, validation, and test sets."""
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            features, labels, test_size=test_size, random_state=random_state, 
            stratify=labels if len(np.unique(labels)) > 1 else None
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=random_state,
            stratify=y_temp if len(np.unique(y_temp)) > 1 else None
        )
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def save_preprocessors(self, model_path=MODEL_PATH):
        """Save label encoder and scaler."""
        os.makedirs(model_path, exist_ok=True)
        
        joblib.dump(self.label_encoder, os.path.join(model_path, 'label_encoder.pkl'))
        joblib.dump(self.scaler, os.path.join(model_path, 'feature_scaler.pkl'))
        
        if hasattr(self, 'feature_names'):
            joblib.dump(self.feature_names, os.path.join(model_path, 'feature_names.pkl'))
        
    def load_preprocessors(self, model_path=MODEL_PATH):
        """Load saved label encoder and scaler."""
        self.label_encoder = joblib.load(os.path.join(model_path, 'label_encoder.pkl'))
        self.scaler = joblib.load(os.path.join(model_path, 'feature_scaler.pkl'))
        
        feature_names_path = os.path.join(model_path, 'feature_names.pkl')
        if os.path.exists(feature_names_path):
            self.feature_names = joblib.load(feature_names_path)
    
    def get_feature_names(self):
        """Get feature names for interpretability."""
        feature_names = []
        
        # MFCC features
        for i in range(N_MFCC):
            feature_names.append(f'mfcc_{i}_mean')
        for i in range(N_MFCC):
            feature_names.append(f'mfcc_{i}_std')
            
        # Chroma features
        for i in range(N_CHROMA):
            feature_names.append(f'chroma_{i}_mean')
        for i in range(N_CHROMA):
            feature_names.append(f'chroma_{i}_std')
            
        # Spectral contrast features
        for i in range(N_SPECTRAL_CONTRAST):
            feature_names.append(f'spectral_contrast_{i}_mean')
        for i in range(N_SPECTRAL_CONTRAST):
            feature_names.append(f'spectral_contrast_{i}_std')
            
        # Additional features
        feature_names.extend(['zero_crossing_rate', 'spectral_centroid', 'spectral_rolloff'])
        
        return feature_names