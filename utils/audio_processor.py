"""Audio processing utilities for feature extraction and preprocessing."""

import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import os
import tempfile
from io import BytesIO
from config import *

class AudioProcessor:
    def __init__(self, sample_rate=SAMPLE_RATE, duration=DURATION):
        self.sample_rate = sample_rate
        self.duration = duration
        self.target_length = sample_rate * duration

    def load_audio(self, file_path):
        """Load audio file and convert to standard format."""
        try:
            if file_path.endswith(('.mp3', '.MP3')):
                # Use in-memory processing to avoid temporary file issues on Windows
                audio = AudioSegment.from_mp3(file_path)
                
                # Convert to raw audio data
                raw_data = audio.raw_data
                frame_rate = audio.frame_rate
                channels = audio.channels
                sample_width = audio.sample_width
                
                # Convert to numpy array
                if sample_width == 1:
                    dtype = np.uint8
                elif sample_width == 2:
                    dtype = np.int16
                elif sample_width == 4:
                    dtype = np.int32
                else:
                    dtype = np.float32
                
                # Convert raw data to numpy array
                audio_array = np.frombuffer(raw_data, dtype=dtype)
                
                # Handle stereo to mono conversion
                if channels == 2:
                    audio_array = audio_array.reshape((-1, 2))
                    audio_array = audio_array.mean(axis=1)
                
                # Normalize to [-1, 1] range
                if dtype == np.uint8:
                    max_val = 255
                    audio_array = (audio_array.astype(np.float32) - 128) / 128
                elif dtype == np.int16:
                    max_val = 32767
                    audio_array = audio_array.astype(np.float32) / max_val
                elif dtype == np.int32:
                    max_val = 2147483647
                    audio_array = audio_array.astype(np.float32) / max_val
                # float32 is already normalized
                
                # Resample if necessary
                if frame_rate != self.sample_rate:
                    audio_array = librosa.resample(audio_array, orig_sr=frame_rate, target_sr=self.sample_rate)
                
                return audio_array, self.sample_rate
                
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
            # For longer files, take the middle section instead of just the beginning
            # This is more representative for music genre classification
            start_sample = (len(y) - self.target_length) // 2
            y = y[start_sample:start_sample + self.target_length]
        
        return y

    def extract_features(self, y):
        """Extract audio features to match the exact trained model format."""
        features = []
        
        # 1. Length
        features.append(float(len(y)))
        
        # 2-3. Chroma STFT (mean and var)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=self.sample_rate)
        features.append(float(np.mean(chroma_stft)))
        features.append(float(np.var(chroma_stft)))
        
        # 4-5. RMS (mean and var)
        rms = librosa.feature.rms(y=y)
        features.append(float(np.mean(rms)))
        features.append(float(np.var(rms)))
        
        # 6-7. Spectral centroid (mean and var)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=self.sample_rate)
        features.append(float(np.mean(spectral_centroid)))
        features.append(float(np.var(spectral_centroid)))
        
        # 8-9. Spectral bandwidth (mean and var)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=self.sample_rate)
        features.append(float(np.mean(spectral_bandwidth)))
        features.append(float(np.var(spectral_bandwidth)))
        
        # 10-11. Rolloff (mean and var)
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=self.sample_rate)
        features.append(float(np.mean(rolloff)))
        features.append(float(np.var(rolloff)))
        
        # 12-13. Zero crossing rate (mean and var)
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(float(np.mean(zcr)))
        features.append(float(np.var(zcr)))
        
        # 14-15. Harmony (mean and var)
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        features.append(float(np.mean(y_harmonic)))
        features.append(float(np.var(y_harmonic)))
        
        # 16-17. Perceptr (mean and var) - using percussive component
        features.append(float(np.mean(y_percussive)))
        features.append(float(np.var(y_percussive)))
        
        # 18. Tempo
        tempo, beats = librosa.beat.beat_track(y=y, sr=self.sample_rate)
        features.append(float(tempo))
        
        # 19-58. MFCCs 1-20 (mean and var for each)
        mfcc = librosa.feature.mfcc(y=y, sr=self.sample_rate, n_mfcc=20)
        for i in range(20):
            features.append(float(np.mean(mfcc[i])))
            features.append(float(np.var(mfcc[i])))
        
        # Convert to numpy array and handle any NaN/inf values
        features_array = np.array(features, dtype=np.float64)
        features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Apply additional normalization to handle feature scale issues
        # Clip extreme values that might bias the model
        features_array = np.clip(features_array, -1e6, 1e6)
        
        return features_array

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