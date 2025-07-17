"""Prediction script for single audio files."""

import argparse
import os
import numpy as np
from utils.audio_processor import AudioProcessor
from utils.data_loader import DataLoader
from models.genre_classifier import GenreClassifier
from config import *

class GenrePredictor:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.data_loader = DataLoader()
        self.classifier = GenreClassifier(input_dim=1)
        
        # Load trained model and preprocessors
        self.load_trained_model()
    
    def load_trained_model(self):
        """Load the trained model and preprocessors."""
        try:
            # Load preprocessors
            self.data_loader.load_preprocessors()
            
            # Load model
            self.classifier.load_model()
            print("✅ Model and preprocessors loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            print("Please train the model first using: python train_model.py")
            raise
    
    def predict_genre(self, audio_file_path):
        """Predict genre for an audio file."""
        try:
            # Extract features
            features, audio_data = self.audio_processor.process_file(audio_file_path)
            
            # Scale features
            features_scaled = self.data_loader.scaler.transform(features.reshape(1, -1))
            
            # Make prediction
            result = self.classifier.predict_single(features_scaled[0], self.data_loader.label_encoder)
            
            return result, audio_data
            
        except Exception as e:
            print(f"❌ Error predicting genre: {str(e)}")
            return None, None

def main():
    parser = argparse.ArgumentParser(description='Predict music genre from audio file')
    parser.add_argument('audio_file', help='Path to audio file')
    parser.add_argument('--show-top', type=int, default=3, help='Number of top predictions to show')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.audio_file):
        print(f"❌ File not found: {args.audio_file}")
        return
    
    # Initialize predictor
    try:
        predictor = GenrePredictor()
    except:
        return
    
    # Make prediction
    print(f"🎵 Analyzing: {args.audio_file}")
    result, audio_data = predictor.predict_genre(args.audio_file)
    
    if result:
        print(f"\n🎯 Predicted Genre: {result['predicted_genre'].title()}")
        print(f"🎲 Confidence: {result['confidence']:.2%}")
        
        print(f"\n🏆 Top {args.show_top} Predictions:")
        for i, (genre, prob) in enumerate(result['top_predictions'][:args.show_top], 1):
            print(f"  {i}. {genre.title()}: {prob:.2%}")
    else:
        print("❌ Failed to analyze the audio file")

if __name__ == "__main__":
    main()