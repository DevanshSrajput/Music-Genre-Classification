"""Training script for the genre classification model."""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

from utils.data_loader import DataLoader
from models.genre_classifier import GenreClassifier
from config import *

def plot_training_history(history, save_path='results/training_history.png'):
    """Plot training history."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, genre_names, save_path='results/confusion_matrix.png'):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=genre_names, yticklabels=genre_names,
                cbar_kws={'label': 'Number of Samples'})
    plt.title('Confusion Matrix - Music Genre Classification', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Genre', fontsize=12)
    plt.ylabel('True Genre', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main training function."""
    print("🎵 Starting Music Genre Classification Training")
    print("=" * 50)
    
    # Initialize data loader
    data_loader = DataLoader(use_precomputed_features=True)
    
    # Load dataset
    print("📂 Loading dataset...")
    
    try:
        features, labels = data_loader.load_dataset()
        print(f"✅ Loaded {len(features)} samples with {features.shape[1]} features")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("Please ensure you have either:")
        print("1. features_30_sec.csv or features_3_sec.csv in the data/ directory")
        print("2. genres_original/ folder with audio files in the data/ directory")
        return
    
    # Prepare data
    print("🔧 Preparing data...")
    features_scaled, labels_encoded = data_loader.prepare_data(features, labels)
    
    # Split data
    print("🔀 Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(
        features_scaled, labels_encoded
    )
    
    print(f"📈 Data splits:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Validation set: {X_val.shape[0]} samples") 
    print(f"  Test set: {X_test.shape[0]} samples")
    
    # Save preprocessors
    print("💾 Saving preprocessors...")
    data_loader.save_preprocessors()
    
    # Initialize model
    print("🧠 Building model...")
    classifier = GenreClassifier(input_dim=X_train.shape[1])
    model = classifier.build_deep_model()
    
    print("📋 Model Summary:")
    model.summary()
    
    # Train model
    print("🚀 Starting training...")
    history = classifier.train(X_train, y_train, X_val, y_val)
    
    # Plot training history
    print("📈 Plotting training history...")
    plot_training_history(history)
    
    # Evaluate on test set
    print("🎯 Evaluating on test set...")
    test_predictions = classifier.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    print(f"🏆 Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Classification report
    genre_names = data_loader.label_encoder.classes_
    print("\n📊 Classification Report:")
    print(classification_report(y_test, test_predictions, target_names=genre_names, digits=4))
    
    # Confusion matrix
    print("📊 Generating confusion matrix...")
    plot_confusion_matrix(y_test, test_predictions, genre_names)
    
    # Save model
    print(f"💾 Saving model to {TRAINED_MODEL_PATH}...")
    classifier.save_model()
    
    print("\n🎉 Training Completed Successfully!")
    print(f"🏆 Final Test Accuracy: {test_accuracy*100:.2f}%")
    print("🚀 Run 'streamlit run app.py' to start the web interface")

if __name__ == "__main__":
    main()