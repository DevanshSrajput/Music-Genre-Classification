"""Audio processing utilities for feature extraction and preprocessing."""

import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import os
import tempfile
from config import *

class AudioProcessor:
    def __init__(self, sample_rate=SAMPLE_RATE, duration=DURATION):
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = sample_rate * duration

    def load_audio(self, file_path):
        """Load audio file and convert to standard format."""
        try:
            if file_path.endswith('.mp3'):
                audio = AudioSegment.from_mp3(file_path)
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    audio.export(temp_file.name, format='wav')
                    y, sr = librosa.load(temp_file.name, sr=self.sample_rate)
                    os.unlink(temp_file.name)
            else:
                y, sr = librosa.load(file_path, sr=self.sample_rate)
            
            return y, sr
        except Exception as e:
            raise Exception(f"Error loading audio file: {str(e)}")

    def preprocess_audio(self, y):
        """Preprocess audio: normalize and handle length."""
        y = librosa.util.normalize(y)
        
        if len(y) < self.target_length:
            y = np.pad(y, (0, self.target_length - len(y)), mode='constant')
        elif len(y) > self.target_length:
            y = y[:self.target_length]
        
        return y

    def extract_features(self, y):
        """Extract audio features: MFCCs, chroma, and spectral contrast."""
        features = {}
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=N_MFCC)
        features['mfcc'] = np.mean(mfccs, axis=1)
        features['mfcc_std'] = np.std(mfccs, axis=1)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=self.sample_rate)
        features['chroma'] = np.mean(chroma, axis=1)
        features['chroma_std'] = np.std(chroma, axis=1)
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=self.sample_rate)
        features['spectral_contrast'] = np.mean(spectral_contrast, axis=1)
        features['spectral_contrast_std'] = np.std(spectral_contrast, axis=1)
        
        # Additional features
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
        features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=self.sample_rate))
        features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=y, sr=self.sample_rate))
        
        # Concatenate all features
        feature_vector = np.concatenate([
            features['mfcc'],
            features['mfcc_std'],
            features['chroma'],
            features['chroma_std'],
            features['spectral_contrast'],
            features['spectral_contrast_std'],
            [features['zero_crossing_rate']],
            [features['spectral_centroid']],
            [features['spectral_rolloff']]
        ])
        
        return feature_vector

    def process_file(self, file_path):
        """Complete pipeline: load, preprocess, and extract features."""
        y, sr = self.load_audio(file_path)
        y_processed = self.preprocess_audio(y)
        features = self.extract_features(y_processed)
        return features, y_processed

    def get_audio_info(self, file_path):
        """Get basic audio file information."""
        try:
            y, sr = self.load_audio(file_path)
            duration = len(y) / sr
            return {
                'duration': duration,
                'sample_rate': sr,
                'channels': 1,
                'samples': len(y)
            }
        except Exception as e:
            return {'error': str(e)}