"""Deep learning models for genre classification."""

import os
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union

# TensorFlow and Keras imports
import tensorflow as tf
try:
    # Try TensorFlow 2.x imports first
    from tensorflow.keras.models import Sequential, Model, load_model  # type: ignore
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input  # type: ignore
    from tensorflow.keras.optimizers import Adam  # type: ignore
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint  # type: ignore
    from tensorflow.keras.utils import to_categorical  # type: ignore
except ImportError:
    # Fallback to standalone Keras
    from keras.models import Sequential, Model, load_model  # type: ignore
    from keras.layers import Dense, Dropout, BatchNormalization , Input  # type: ignore
    from keras.optimizers import Adam  # type: ignore
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint  # type: ignore
    from keras.utils import to_categorical  # type: ignore

# Config imports with fallback
try:
    from config import *
except ImportError:
    # Fallback values if config is not available
    GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    EPOCHS = 100
    BATCH_SIZE = 32
    TRAINED_MODEL_PATH = 'models/trained_model.h5'

class GenreClassifier:
    def __init__(self, input_dim: int, num_classes: int = len(GENRES)):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.model: Optional[tf.keras.Model] = None
        
    def build_mlp_model(self):
        """Build a Multi-Layer Perceptron model."""
        model = Sequential([
            Input(shape=(self.input_dim,)),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(64, activation='relu'),
            Dropout(0.2),
            
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',  # Use string instead
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def build_deep_model(self):
        """Build a deeper neural network model."""
        model = Sequential([
            Input(shape=(self.input_dim,)),
            Dense(1024, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(64, activation='relu'),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.1),
            
            Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',  # Use string instead
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def get_callbacks(self, model_path, early_stopping_patience=25, use_early_stopping=True):
        """Get training callbacks."""
        callbacks = []
        
        # Always add ModelCheckpoint
        callbacks.append(ModelCheckpoint(
            filepath=model_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ))
        
        # Always add ReduceLROnPlateau
        callbacks.append(ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-7,  # type: ignore
            verbose=1
        ))
        
        # Add EarlyStopping only if requested
        if use_early_stopping:
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ))
        
        return callbacks
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, 
              epochs: int = EPOCHS, batch_size: int = BATCH_SIZE, 
              model_type: str = 'mlp', early_stopping_patience: int = 25, use_early_stopping: bool = True):
        """Train the model."""
        if self.model is None:
            try:
                if model_type == 'deep':
                    self.build_deep_model()
                else:
                    self.build_mlp_model()
            except Exception as e:
                raise ValueError(f"Error building {model_type} model: {str(e)}")
        
        # Safety check
        if self.model is None:
            raise ValueError("Model failed to build. Check your build methods.")
        
        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, num_classes=self.num_classes)
        y_val_cat = to_categorical(y_val, num_classes=self.num_classes)
        
        # Get callbacks
        callbacks = self.get_callbacks(TRAINED_MODEL_PATH, early_stopping_patience, use_early_stopping)
        
        # Train the model
        history = self.model.fit(  # type: ignore
            X_train, y_train_cat,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val_cat),
            callbacks=callbacks,
            verbose='auto'
        )
        
        return history
    
    def predict(self, X: np.ndarray, return_probabilities: bool = False):
        """Make predictions."""
        if self.model is None:
            raise ValueError("Model is not trained or loaded. Please train or load a model first.")
            
        predictions = self.model.predict(X)  # type: ignore
        
        if return_probabilities:
            return predictions
        else:
            return np.argmax(predictions, axis=1)
    
    def predict_single(self, feature_vector: np.ndarray, label_encoder) -> Dict[str, Any]:
        """Predict genre for a single audio file."""
        if self.model is None:
            raise ValueError("Model is not trained or loaded. Please train or load a model first.")
            
        # Reshape for prediction
        feature_vector = feature_vector.reshape(1, -1)
        
        # Get probabilities
        probabilities = self.model.predict(feature_vector, verbose='auto')[0]  # type: ignore
        
        # Get predicted class
        predicted_class = np.argmax(probabilities)
        predicted_genre = label_encoder.inverse_transform([predicted_class])[0]
        confidence = probabilities[predicted_class]
        
        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[::-1][:3]
        top_genres = [(label_encoder.inverse_transform([idx])[0], probabilities[idx]) 
                     for idx in top_indices]
        
        return {
            'predicted_genre': predicted_genre,
            'confidence': float(confidence),
            'top_predictions': top_genres,
            'all_probabilities': dict(zip(label_encoder.classes_, probabilities))
        }
    
    def load_model(self, model_path: str = TRAINED_MODEL_PATH):
        """Load a trained model."""
        self.model = load_model(model_path)  # type: ignore
        
    def save_model(self, model_path: str = TRAINED_MODEL_PATH):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("No model to save. Please train or load a model first.")
            
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)  # type: ignore