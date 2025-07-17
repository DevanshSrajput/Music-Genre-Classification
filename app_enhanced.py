"""Enhanced Streamlit web application for Music Genre Classification with Training Capabilities."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import librosa
import pandas as pd
from io import BytesIO
import tempfile
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from typing import Dict, Any, Optional, List, Tuple

from predict import GenrePredictor
from utils.audio_processor import AudioProcessor
from utils.data_loader import DataLoader
from models.genre_classifier import GenreClassifier
from config import *

# Page configuration
st.set_page_config(
    page_title="🎵 Music Genre Classifier Pro",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .confidence-high { color: #28a745; }
    .confidence-medium { color: #ffc107; }
    .confidence-low { color: #dc3545; }
    .training-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e9ecef;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .feature-section {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'training_in_progress' not in st.session_state:
    st.session_state.training_in_progress = False
if 'training_results' not in st.session_state:
    st.session_state.training_results = None
if 'current_tab' not in st.session_state:
    st.session_state.current_tab = "Prediction"

@st.cache_resource
def load_predictor():
    """Load the trained model (cached)."""
    try:
        return GenrePredictor()
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        st.info("Please ensure the model is trained or train a new model using the Training tab.")
        return None

def train_model_with_progress(epochs, early_stopping_patience, use_early_stopping, data_path):
    """Train model with real-time progress updates."""
    try:
        # Initialize progress tracking
        st.session_state.training_progress = {
            'current_epoch': 0,
            'total_epochs': epochs,
            'current_loss': 0.0,
            'current_accuracy': 0.0,
            'val_loss': 0.0,
            'val_accuracy': 0.0,
            'status': 'Starting...'
        }
        
        # Load data using proper DataLoader workflow
        st.session_state.training_progress['status'] = 'Loading data...'
        data_loader = DataLoader()
        
        # Load dataset from CSV
        features, labels = data_loader.load_dataset(features_csv=data_path)
        
        # Prepare data (clean, encode, scale)
        st.session_state.training_progress['status'] = 'Preparing data...'
        features_scaled, labels_encoded = data_loader.prepare_data(features, labels)
        
        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = data_loader.split_data(
            features_scaled, labels_encoded
        )
        
        # Save preprocessors
        data_loader.save_preprocessors()
        
        # Create and train model
        st.session_state.training_progress['status'] = 'Building model...'
        classifier = ImprovedGenreClassifier(input_dim=X_train.shape[1])
        classifier.build_improved_model()
        
        # Ensure model was built successfully
        if classifier.model is None:
            raise ValueError("Failed to build model")
        
        # Train with custom parameters and progress callback
        st.session_state.training_progress['status'] = 'Training model...'
        history = classifier.train_with_progress(
            X_train, y_train, X_val, y_val,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            use_early_stopping=use_early_stopping
        )
        
        # Evaluate model - convert test labels to categorical for evaluation
        st.session_state.training_progress['status'] = 'Evaluating model...'
        from tensorflow.keras.utils import to_categorical
        y_test_cat = to_categorical(y_test, num_classes=len(GENRES))
        test_loss, test_accuracy = classifier.model.evaluate(X_test, y_test_cat, verbose=0)
        
        # Get predictions for detailed metrics
        y_pred = classifier.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = y_test  # Keep original integer labels for metrics
        
        # Store results with percentage accuracy
        st.session_state.training_results = {
            'history': history,
            'test_accuracy': test_accuracy,
            'test_accuracy_percentage': test_accuracy * 100,  # Convert to percentage
            'test_loss': test_loss,
            'y_true': y_true_classes,
            'y_pred': y_pred_classes,
            'y_pred_proba': y_pred,
            'model': classifier,
            'total_params': classifier.model.count_params()
        }
        
        # Update final status with percentage
        st.session_state.training_progress['status'] = f'Training completed! Final Test Accuracy: {test_accuracy * 100:.2f}%'
        st.session_state.training_progress['final_test_accuracy'] = test_accuracy * 100
        
    except Exception as e:
        st.session_state.training_results = {'error': str(e)}
        st.session_state.training_progress['status'] = f'Error: {str(e)}'
    finally:
        st.session_state.training_in_progress = False


class ImprovedGenreClassifier(GenreClassifier):
    """Enhanced GenreClassifier with better architecture and progress tracking."""
    
    def build_improved_model(self):
        """Build an improved model with better architecture for higher accuracy."""
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
        from tensorflow.keras.optimizers import Adam
        from tensorflow.keras.regularizers import l2
        
        # Enhanced architecture for better performance
        model = Sequential([
            Input(shape=(self.input_dim,)),
            
            # First layer - larger capacity
            Dense(1024, activation='relu', kernel_regularizer=l2(0.00001)),
            BatchNormalization(),
            Dropout(0.4),
            
            # Second layer - maintain capacity
            Dense(512, activation='relu', kernel_regularizer=l2(0.00001)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Third layer - feature refinement
            Dense(256, activation='relu', kernel_regularizer=l2(0.00001)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Fourth layer - pattern recognition
            Dense(128, activation='relu', kernel_regularizer=l2(0.00001)),
            BatchNormalization(),
            Dropout(0.2),
            
            # Fifth layer - final feature extraction
            Dense(64, activation='relu', kernel_regularizer=l2(0.00001)),
            BatchNormalization(),
            Dropout(0.2),
            
            # Sixth layer - classification preparation
            Dense(32, activation='relu'),
            Dropout(0.1),
            
            # Output layer
            Dense(self.num_classes, activation='softmax')
        ])
        
        # Use Adam optimizer with optimized learning rate
        optimizer = Adam(
            learning_rate=0.001,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_with_progress(self, X_train, y_train, X_val, y_val, epochs=100, 
                           early_stopping_patience=25, use_early_stopping=True):
        """Train model with Streamlit-compatible progress updates."""
        from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback
        from tensorflow.keras.utils import to_categorical
        import gc
        
        # Custom callback for enhanced real-time progress tracking
        class StreamlitProgressCallback(Callback):
            def __init__(self, total_epochs):
                super().__init__()
                self.total_epochs = total_epochs
                self.best_val_accuracy = 0.0
                self.best_train_accuracy = 0.0
                self.improvement_count = 0
                self.epochs_since_improvement = 0
                
            def on_epoch_begin(self, epoch, logs=None):
                # Update status at the beginning of each epoch
                if 'training_progress' in st.session_state:
                    st.session_state.training_progress['status'] = f'🚀 Training Epoch {epoch + 1}/{self.total_epochs}... Processing batches...'
                    st.session_state.training_progress['current_epoch'] = epoch
                
            def on_epoch_end(self, epoch, logs=None):
                if logs is None:
                    logs = {}
                    
                # Calculate percentage accuracies
                train_acc = logs.get('accuracy', 0.0)
                val_acc = logs.get('val_accuracy', 0.0)
                train_loss = logs.get('loss', 0.0)
                val_loss = logs.get('val_loss', 0.0)
                
                train_acc_pct = train_acc * 100
                val_acc_pct = val_acc * 100
                
                # Track improvements
                if val_acc > self.best_val_accuracy:
                    self.best_val_accuracy = val_acc
                    self.improvement_count += 1
                    self.epochs_since_improvement = 0
                    improvement_indicator = "📈 NEW BEST!"
                else:
                    self.epochs_since_improvement += 1
                    improvement_indicator = f"📊 (Best: {self.best_val_accuracy * 100:.2f}%)"
                
                if train_acc > self.best_train_accuracy:
                    self.best_train_accuracy = train_acc
                
                # Calculate progress percentage
                progress_pct = ((epoch + 1) / self.total_epochs) * 100
                
                # Update progress in session state with detailed information
                if 'training_progress' in st.session_state:
                    st.session_state.training_progress.update({
                        'current_epoch': epoch + 1,
                        'total_epochs': self.total_epochs,
                        'current_loss': float(train_loss),
                        'current_accuracy': float(train_acc),
                        'val_loss': float(val_loss),
                        'val_accuracy': float(val_acc),
                        'current_accuracy_pct': train_acc_pct,
                        'val_accuracy_pct': val_acc_pct,
                        'best_val_accuracy': self.best_val_accuracy * 100,
                        'best_train_accuracy': self.best_train_accuracy * 100,
                        'improvement_count': self.improvement_count,
                        'epochs_since_improvement': self.epochs_since_improvement,
                        'progress_percentage': progress_pct,
                        'improvement_indicator': improvement_indicator,
                        'status': f'✅ Epoch {epoch + 1}/{self.total_epochs} Complete | Train: {train_acc_pct:.2f}% | Val: {val_acc_pct:.2f}% | {improvement_indicator}'
                    })
                
                # Force garbage collection periodically
                if (epoch + 1) % 10 == 0:
                    gc.collect()
                    
            def on_batch_end(self, batch, logs=None):
                # Update status during batch processing for real-time feedback
                if batch % 10 == 0 and 'training_progress' in st.session_state:
                    current_epoch = st.session_state.training_progress.get('current_epoch', 0)
                    st.session_state.training_progress['status'] = f'🔄 Epoch {current_epoch + 1}/{self.total_epochs} - Processing batch {batch + 1}...'
        
        # Data augmentation for better generalization
        from sklearn.utils import shuffle
        
        # Add some noise to training data for regularization
        noise_factor = 0.01
        X_train_augmented = X_train + noise_factor * np.random.normal(0, 1, X_train.shape)
        X_train_combined = np.vstack([X_train, X_train_augmented])
        y_train_combined = np.vstack([y_train_cat, y_train_cat])
        
        # Shuffle the combined data
        X_train_combined, y_train_combined = shuffle(X_train_combined, y_train_combined, random_state=42)
        
        # Convert labels to categorical
        y_val_cat = to_categorical(y_val, num_classes=self.num_classes)
        
        # Setup callbacks
        callbacks = [StreamlitProgressCallback(epochs)]
        
        # Model checkpoint
        callbacks.append(ModelCheckpoint(
            filepath=TRAINED_MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=0
        ))
        
        # Learning rate reduction
        callbacks.append(ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=0
        ))
        
        # Early stopping if requested
        if use_early_stopping:
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=0
            ))
        
        # Train the model with optimized parameters and augmented data
        history = self.model.fit(
            X_train_combined, y_train_combined,
            batch_size=8,  # Smaller batch size for better convergence
            epochs=epochs,
            validation_data=(X_val, y_val_cat),
            callbacks=callbacks,
            verbose=0,
            shuffle=True,
            class_weight='balanced'  # Handle class imbalance
        )
        
        return history

def display_training_results():
    """Display comprehensive training results."""
    if st.session_state.training_results is None:
        return
    
    results = st.session_state.training_results
    
    if 'error' in results:
        st.error(f"Training failed: {results['error']}")
        return
    
    st.success(f"🎉 Training completed successfully! Final Test Accuracy: {results.get('test_accuracy_percentage', results['test_accuracy'] * 100):.2f}%")
    
    # Training metrics with percentage display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        test_acc_pct = results.get('test_accuracy_percentage', results['test_accuracy'] * 100)
        st.metric("Final Test Accuracy", f"{test_acc_pct:.2f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Final Test Loss", f"{results['test_loss']:.4f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        epochs_trained = len(results['history'].history['loss'])
        st.metric("Epochs Trained", epochs_trained)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Parameters", f"{results['total_params']:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Training history plots
    col1, col2 = st.columns(2)
    
    with col1:
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(
            y=results['history'].history['accuracy'],
            mode='lines',
            name='Training Accuracy',
            line=dict(color='blue', width=2)
        ))
        fig_acc.add_trace(go.Scatter(
            y=results['history'].history['val_accuracy'],
            mode='lines',
            name='Validation Accuracy',
            line=dict(color='red', width=2)
        ))
        fig_acc.update_layout(
            title='🎯 Training & Validation Accuracy',
            xaxis_title='Epoch',
            yaxis_title='Accuracy',
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig_acc, use_container_width=True)
    
    with col2:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(
            y=results['history'].history['loss'],
            mode='lines',
            name='Training Loss',
            line=dict(color='blue', width=2)
        ))
        fig_loss.add_trace(go.Scatter(
            y=results['history'].history['val_loss'],
            mode='lines',
            name='Validation Loss',
            line=dict(color='red', width=2)
        ))
        fig_loss.update_layout(
            title='📉 Training & Validation Loss',
            xaxis_title='Epoch',
            yaxis_title='Loss',
            hovermode='x unified',
            template='plotly_white'
        )
        st.plotly_chart(fig_loss, use_container_width=True)
    
    # Confusion Matrix
    st.subheader("📊 Confusion Matrix")
    cm = confusion_matrix(results['y_true'], results['y_pred'])
    
    fig_cm = px.imshow(
        cm,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=GENRES,
        y=GENRES,
        color_continuous_scale='Blues',
        title="Confusion Matrix - Model Performance by Genre",
        aspect="auto"
    )
    fig_cm.update_xaxes(side="top")
    fig_cm.update_layout(template='plotly_white')
    st.plotly_chart(fig_cm, use_container_width=True)
    
    # Classification Report
    st.subheader("📈 Classification Report")
    report = classification_report(
        results['y_true'], 
        results['y_pred'], 
        target_names=GENRES,
        output_dict=True
    )
    
    # Convert to DataFrame for better display
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.round(3), use_container_width=True)
    
    # Per-class performance
    st.subheader("🎼 Per-Genre Performance")
    genre_metrics = []
    for i, genre in enumerate(GENRES):
        if genre in report:
            genre_report = report[genre]  # type: ignore
            precision = genre_report['precision']  # type: ignore
            recall = genre_report['recall']  # type: ignore
            f1 = genre_report['f1-score']  # type: ignore
            support = genre_report['support']  # type: ignore
            genre_metrics.append({
                'Genre': genre.title(),
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Support': int(support)
            })
    
    if genre_metrics:
        metrics_df = pd.DataFrame(genre_metrics)
        
        # Create bar chart for F1-scores
        fig_f1 = px.bar(
            metrics_df, 
            x='Genre', 
            y='F1-Score',
            title='F1-Score by Genre',
            color='F1-Score',
            color_continuous_scale='RdYlBu_r'
        )
        fig_f1.update_layout(template='plotly_white')
        st.plotly_chart(fig_f1, use_container_width=True)

def plot_waveform(audio_data, sample_rate):
    """Plot audio waveform."""
    time = np.linspace(0, len(audio_data) / sample_rate, len(audio_data))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time, y=audio_data,
        mode='lines',
        name='Waveform',
        line=dict(color='#1f77b4', width=1)
    ))
    
    fig.update_layout(
        title="🌊 Audio Waveform",
        xaxis_title="Time (seconds)",
        yaxis_title="Amplitude",
        height=300,
        showlegend=False,
        template='plotly_white'
    )
    
    return fig

def plot_spectrogram(audio_data, sample_rate):
    """Plot spectrogram."""
    D = librosa.amplitude_to_db(np.abs(librosa.stft(audio_data)), ref=np.max)
    
    fig = go.Figure(data=go.Heatmap(
        z=D,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title="dB")
    ))
    
    fig.update_layout(
        title="🎵 Spectrogram",
        xaxis_title="Time Frames",
        yaxis_title="Frequency Bins",
        height=400,
        template='plotly_white'
    )
    
    return fig

def plot_audio_features(features):
    """Plot extracted audio features."""
    # Create a radar chart for key features
    feature_names = ['MFCC Mean', 'Chroma Mean', 'Spectral Contrast Mean', 
                    'ZCR', 'Spectral Centroid', 'Spectral Rolloff']
    
    # Normalize features for radar chart (take first few features)
    feature_values = features[:6]
    normalized_values = (feature_values - np.min(feature_values)) / (np.max(feature_values) - np.min(feature_values))
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=normalized_values,
        theta=feature_names,
        fill='toself',
        name='Audio Features'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True,
        title="🎛️ Audio Feature Profile"
    )
    
    return fig

def plot_predictions(predictions):
    """Plot prediction probabilities."""
    genres = list(predictions.keys())
    probabilities = [predictions[genre] for genre in genres]
    
    # Sort by probability
    sorted_data = sorted(zip(genres, probabilities), key=lambda x: x[1], reverse=True)
    genres_sorted, probs_sorted = zip(*sorted_data)
    
    # Color coding based on probability
    colors = []
    for prob in probs_sorted:
        if prob > 0.6:
            colors.append('#28a745')  # High confidence - green
        elif prob > 0.3:
            colors.append('#ffc107')  # Medium confidence - yellow
        else:
            colors.append('#dc3545')  # Low confidence - red
    
    fig = px.bar(
        x=probs_sorted,
        y=genres_sorted,
        orientation='h',
        title="🎯 Genre Prediction Probabilities",
        labels={'x': 'Probability', 'y': 'Genre'},
        color_discrete_sequence=colors
    )
    
    fig.update_layout(
        height=400,
        showlegend=False,
        yaxis={'categoryorder': 'total ascending'},
        template='plotly_white'
    )
    
    return fig

def get_genre_info(genre):
    """Get comprehensive information about a music genre."""
    genre_info = {
        'blues': {
            'description': 'Blues is a music genre characterized by the use of blue notes, twelve-bar progression, and call-and-response vocals.',
            'characteristics': ['Blue notes', 'Twelve-bar progression', 'Call-and-response vocals', 'Expressive vocals'],
            'artists': ['B.B. King', 'Muddy Waters', 'Robert Johnson', 'Stevie Ray Vaughan'],
            'origin': 'African-American communities in the Deep South (1860s)',
            'tempo': '60-120 BPM',
            'key_instruments': ['Guitar', 'Harmonica', 'Piano', 'Bass']
        },
        'classical': {
            'description': 'Classical music is art music produced in Western musical tradition, known for its complex musical structures.',
            'characteristics': ['Complex harmonies', 'Orchestral arrangements', 'Formal structures', 'Dynamic contrasts'],
            'artists': ['Bach', 'Mozart', 'Beethoven', 'Chopin'],
            'origin': 'Europe (11th century)',
            'tempo': '40-200 BPM (varies widely)',
            'key_instruments': ['Orchestra', 'Piano', 'Violin', 'Cello']
        },
        'country': {
            'description': 'Country music is a genre that originated in rural areas of the Southern United States.',
            'characteristics': ['Storytelling lyrics', 'Twangy vocals', 'Simple chord progressions', 'Rural themes'],
            'artists': ['Johnny Cash', 'Dolly Parton', 'Willie Nelson', 'Hank Williams'],
            'origin': 'Rural Southern United States (1920s)',
            'tempo': '80-140 BPM',
            'key_instruments': ['Acoustic Guitar', 'Banjo', 'Fiddle', 'Steel Guitar']
        },
        'disco': {
            'description': 'Disco is a genre of dance music containing elements of funk, soul, pop, and salsa.',
            'characteristics': ['Four-on-the-floor beat', 'String sections', 'Catchy melodies', 'Danceable rhythm'],
            'artists': ['Bee Gees', 'ABBA', 'Donna Summer', 'Chic'],
            'origin': 'United States (early 1970s)',
            'tempo': '110-130 BPM',
            'key_instruments': ['Bass Guitar', 'Drums', 'Strings', 'Synthesizer']
        },
        'hiphop': {
            'description': 'Hip hop is a genre characterized by rhythmic spoken lyrics over backing beats.',
            'characteristics': ['Rap vocals', 'Sampling', 'Breakbeats', 'DJ techniques'],
            'artists': ['Tupac', 'Jay-Z', 'Nas', 'Eminem'],
            'origin': 'Bronx, New York (1970s)',
            'tempo': '70-140 BPM',
            'key_instruments': ['Turntables', 'Drum Machine', 'Microphone', 'Sampler']
        },
        'jazz': {
            'description': 'Jazz is a music genre characterized by swing, blue notes, complex chords, and improvisation.',
            'characteristics': ['Improvisation', 'Swing rhythm', 'Blue notes', 'Complex harmonies'],
            'artists': ['Miles Davis', 'John Coltrane', 'Louis Armstrong', 'Duke Ellington'],
            'origin': 'New Orleans (late 19th century)',
            'tempo': '60-200 BPM (varies)',
            'key_instruments': ['Trumpet', 'Saxophone', 'Piano', 'Double Bass']
        },
        'metal': {
            'description': 'Metal is characterized by heavily distorted guitars, emphatic rhythms, and aggressive vocals.',
            'characteristics': ['Distorted guitars', 'Aggressive vocals', 'Fast tempo', 'Power chords'],
            'artists': ['Metallica', 'Iron Maiden', 'Black Sabbath', 'Slayer'],
            'origin': 'United Kingdom (late 1960s)',
            'tempo': '80-200+ BPM',
            'key_instruments': ['Electric Guitar', 'Bass Guitar', 'Drums', 'Vocals']
        },
        'pop': {
            'description': 'Pop music is a genre that emphasizes catchy melodies and mass appeal.',
            'characteristics': ['Catchy melodies', 'Verse-chorus structure', 'Commercial appeal', 'Accessible lyrics'],
            'artists': ['Michael Jackson', 'Madonna', 'Taylor Swift', 'Ariana Grande'],
            'origin': 'United States and United Kingdom (1950s)',
            'tempo': '100-130 BPM',
            'key_instruments': ['Vocals', 'Guitar', 'Keyboards', 'Drums']
        },
        'reggae': {
            'description': 'Reggae is a music genre that originated in Jamaica, characterized by offbeat rhythms.',
            'characteristics': ['Offbeat emphasis', 'Slow tempo', 'Bass-heavy', 'Rastafarian themes'],
            'artists': ['Bob Marley', 'Jimmy Cliff', 'Peter Tosh', 'Burning Spear'],
            'origin': 'Jamaica (late 1960s)',
            'tempo': '60-90 BPM',
            'key_instruments': ['Bass Guitar', 'Drums', 'Guitar', 'Keyboards']
        },
        'rock': {
            'description': 'Rock music is characterized by a strong beat, simple chord progressions, and electric guitars.',
            'characteristics': ['Strong beat', 'Electric guitars', 'Power chords', 'Energetic vocals'],
            'artists': ['The Beatles', 'Led Zeppelin', 'Queen', 'The Rolling Stones'],
            'origin': 'United States (1950s)',
            'tempo': '90-150 BPM',
            'key_instruments': ['Electric Guitar', 'Bass Guitar', 'Drums', 'Vocals']
        }
    }
    
    return genre_info.get(genre, {
        'description': 'No information available', 
        'characteristics': [], 
        'artists': [],
        'origin': 'Unknown',
        'tempo': 'Variable',
        'key_instruments': []
    })

def training_tab():
    """Training interface tab with real-time progress tracking."""
    st.header("🏋️ Model Training")
    
    # Training configuration
    st.markdown('<div class="training-section">', unsafe_allow_html=True)
    st.subheader("⚙️ Training Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        epochs = st.number_input(
            "Number of Epochs",
            min_value=10,
            max_value=500,
            value=100,
            step=10,
            help="Number of training epochs (100+ recommended for full training)"
        )
    
    with col2:
        early_stopping_patience = st.number_input(
            "Early Stopping Patience",
            min_value=5,
            max_value=50,
            value=25,
            step=5,
            help="Number of epochs to wait for improvement before stopping"
        )
    
    with col3:
        use_early_stopping = st.checkbox(
            "Enable Early Stopping",
            value=True,
            help="Uncheck to train for full epochs regardless of validation performance"
        )
    
    # Data configuration
    st.subheader("📊 Data Configuration")
    data_options = {
        "30-second features": "data/features_30_sec.csv",
        "3-second features": "data/features_3_sec.csv"
    }
    
    selected_data = st.selectbox(
        "Select training data",
        options=list(data_options.keys()),
        index=0,
        help="Choose which feature set to use for training"
    )
    
    if selected_data is None:
        selected_data = list(data_options.keys())[0]
    
    data_path = data_options[selected_data]
    
    # Display data info
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Features", len(df.columns) - 1)  # -1 for label column
        with col3:
            st.metric("Genres", df['label'].nunique())
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Training progress display
    if st.session_state.training_in_progress:
        st.subheader("🚀 Training in Progress")
        
        # Add a visual indicator that training is happening
        st.markdown("""
        <div style="background-color: #e8f5e8; padding: 15px; border-radius: 10px; border-left: 5px solid #28a745; margin: 10px 0;">
            <h4 style="color: #155724; margin: 0;">🔥 Model Training Active</h4>
            <p style="color: #155724; margin: 5px 0;">The neural network is being trained on your dataset. This process may take several minutes depending on the number of epochs and dataset size.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize progress tracking if not exists
        if 'training_progress' not in st.session_state:
            st.session_state.training_progress = {
                'current_epoch': 0,
                'total_epochs': epochs,
                'current_loss': 0.0,
                'current_accuracy': 0.0,
                'val_loss': 0.0,
                'val_accuracy': 0.0,
                'status': 'Starting...'
            }
        
        progress = st.session_state.training_progress
        
        # Progress bar with percentage
        if progress['total_epochs'] > 0:
            progress_percent = min(progress['current_epoch'] / progress['total_epochs'], 1.0)
            st.progress(progress_percent)
            st.caption(f"Progress: {progress_percent * 100:.1f}% ({progress['current_epoch']}/{progress['total_epochs']} epochs)")
        
        # Status text with enhanced styling
        st.markdown(f"**Status:** {progress['status']}")
        
        # Training metrics with enhanced real-time display
        if progress['current_epoch'] > 0:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Epoch", f"{progress['current_epoch']}/{progress['total_epochs']}")
            with col2:
                train_acc_pct = progress.get('current_accuracy_pct', progress['current_accuracy'] * 100)
                delta_train = train_acc_pct - progress.get('best_train_accuracy', 0) if 'best_train_accuracy' in progress else None
                st.metric("Training Accuracy", f"{train_acc_pct:.2f}%", delta=f"{delta_train:.2f}%" if delta_train else None)
            with col3:
                val_acc_pct = progress.get('val_accuracy_pct', progress['val_accuracy'] * 100)
                best_val = progress.get('best_val_accuracy', 0)
                delta_val = val_acc_pct - best_val if best_val > 0 else None
                st.metric("Validation Accuracy", f"{val_acc_pct:.2f}%", delta=f"{delta_val:.2f}%" if delta_val else None)
            with col4:
                st.metric("Training Loss", f"{progress['current_loss']:.4f}")
            
            # Additional progress metrics
            col5, col6, col7, col8 = st.columns(4)
            with col5:
                if 'best_val_accuracy' in progress:
                    st.metric("Best Val Accuracy", f"{progress['best_val_accuracy']:.2f}%")
            with col6:
                if 'improvement_count' in progress:
                    st.metric("Improvements", progress['improvement_count'])
            with col7:
                if 'epochs_since_improvement' in progress:
                    st.metric("Since Last Improvement", f"{progress['epochs_since_improvement']} epochs")
            with col8:
                if 'progress_percentage' in progress:
                    st.metric("Overall Progress", f"{progress['progress_percentage']:.1f}%")
            
            # Show improvement indicator
            if 'improvement_indicator' in progress:
                if "NEW BEST" in progress['improvement_indicator']:
                    st.success(progress['improvement_indicator'])
                else:
                    st.info(progress['improvement_indicator'])
        
        # Show final test accuracy if available
        if 'final_test_accuracy' in progress:
            st.success(f"🎉 Final Test Accuracy: {progress['final_test_accuracy']:.2f}%")
        
        # Auto-refresh for real-time updates
        if st.session_state.training_in_progress:
            # Add a container for auto-refresh
            placeholder = st.empty()
            with placeholder.container():
                st.info("🔄 Auto-refreshing every 3 seconds for real-time updates...")
                import time
                time.sleep(3)
                st.experimental_rerun()
        
        # Manual refresh button
        if st.button("🔄 Refresh Progress", key="refresh_progress"):
            st.experimental_rerun()
        
        # Stop training button
        if st.button("🛑 Stop Training", type="secondary", key="stop_training"):
            st.session_state.training_in_progress = False
            st.session_state.training_progress = None
            st.experimental_rerun()
    
    else:
        # Training button with enhanced styling
        st.markdown("---")
        st.subheader("🎯 Ready to Train")
        st.markdown("""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 10px; border-left: 5px solid #007bff; margin: 10px 0;">
            <p style="color: #495057; margin: 0;">Click the button below to start training your music genre classification model. The training process will show real-time progress and metrics.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Start Training", type="primary", use_container_width=True):
            if not os.path.exists(data_path):
                st.error(f"Data file not found: {data_path}")
                return
            
            # Show immediate feedback
            st.success("🚀 Training started! Please wait while the model trains...")
            
            st.session_state.training_in_progress = True
            st.session_state.training_results = None
            
            # Start training directly (no threading for better Streamlit integration)
            train_model_with_progress(epochs, early_stopping_patience, use_early_stopping, data_path)
            
            st.experimental_rerun()
    
    # Display previous training results
    if st.session_state.training_results is not None:
        st.header("📈 Training Results")
        display_training_results()

def prediction_tab():
    """Prediction interface tab."""
    st.header("🎯 Music Genre Prediction")
    
    # Load predictor
    predictor = load_predictor()
    if predictor is None:
        st.warning("⚠️ No trained model found. Please train a model first.")
        return
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=SUPPORTED_FORMATS,
        help=f"Maximum file size: {MAX_FILE_SIZE}MB"
    )
    
    if uploaded_file is not None:
        # File size check
        if uploaded_file.size > MAX_FILE_SIZE * 1024 * 1024:
            st.error(f"File size exceeds {MAX_FILE_SIZE}MB limit")
            return
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            # Audio info
            st.info(f"📁 **File:** {uploaded_file.name} ({uploaded_file.size / 1024 / 1024:.2f} MB)")
            
            # Predict
            with st.spinner("🎵 Analyzing audio..."):
                result, audio_data = predictor.predict_genre(tmp_file_path)
            
            if result:
                # Main prediction result
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                confidence = result['confidence']
                confidence_class = 'confidence-high' if confidence > 0.7 else 'confidence-medium' if confidence > 0.4 else 'confidence-low'
                
                st.markdown(f"""
                ### 🎼 Predicted Genre: **{result['predicted_genre'].title()}**
                <p class="{confidence_class}">**Confidence:** {confidence:.1%}</p>
                """, unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Detailed probabilities
                st.subheader("📊 All Genre Probabilities")
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.write("**Top 5 Predictions:**")
                    if 'top_predictions' in result:
                        for i, (genre, prob) in enumerate(result['top_predictions'][:5], 1):
                            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "🎵"
                            st.write(f"{emoji} **{genre.title()}**: {prob:.1%}")
                    else:
                        # Fallback if top_predictions not available
                        st.write(f"🥇 **{result['predicted_genre'].title()}**: {confidence:.1%}")
                
                with col2:
                    # Create predictions dict if not available
                    if 'all_probabilities' in result:
                        predictions_dict = result['all_probabilities']
                    else:
                        predictions_dict = {result['predicted_genre']: confidence}
                    
                    # Prediction chart
                    st.plotly_chart(
                        plot_predictions(predictions_dict),
                        use_container_width=True
                    )
                
                # Extract and display features
                audio_processor = AudioProcessor()
                try:
                    features, _ = audio_processor.process_file(tmp_file_path)
                except Exception as e:
                    st.error(f"Error extracting features: {str(e)}")
                    features = np.array([])
                
                # Audio visualizations
                st.header("📈 Audio Analysis")
                
                if audio_data is not None:
                    # Waveform and Spectrogram
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(
                            plot_waveform(audio_data, SAMPLE_RATE),
                            use_container_width=True
                        )
                    
                    with col2:
                        st.plotly_chart(
                            plot_spectrogram(audio_data, SAMPLE_RATE),
                            use_container_width=True
                        )
                else:
                    st.warning("Audio data not available for visualization")
                
                # Feature analysis
                col1, col2 = st.columns(2)
                with col1:
                    # Audio features radar chart
                    if len(features) >= 6:
                        st.plotly_chart(
                            plot_audio_features(features),
                            use_container_width=True
                        )
                    else:
                        st.info("Not enough features for radar chart visualization")
                
                with col2:
                    # Audio statistics
                    st.markdown('<div class="feature-section">', unsafe_allow_html=True)
                    st.subheader("🎛️ Audio Statistics")
                    
                    if audio_data is not None:
                        duration = len(audio_data) / SAMPLE_RATE
                        max_amplitude = np.max(np.abs(audio_data))
                        zero_crossings = np.sum(librosa.zero_crossings(audio_data))
                        
                        st.write(f"**Duration:** {duration:.2f} seconds")
                        st.write(f"**Sample Rate:** {SAMPLE_RATE} Hz")
                        st.write(f"**Max Amplitude:** {max_amplitude:.3f}")
                        st.write(f"**Zero Crossings:** {zero_crossings}")
                        st.write(f"**Features Extracted:** {len(features)}")
                    else:
                        st.write("Audio data not available for analysis")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Genre information
                st.header("📚 Genre Information")
                predicted_genre = result['predicted_genre']
                genre_info = get_genre_info(predicted_genre)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader(f"About {predicted_genre.title()}")
                    st.write(genre_info['description'])
                    
                    st.subheader("🎵 Key Characteristics")
                    for char in genre_info['characteristics']:
                        st.write(f"• {char}")
                    
                    st.subheader("📍 Origin & Details")
                    st.write(f"**Origin:** {genre_info['origin']}")
                    st.write(f"**Typical Tempo:** {genre_info['tempo']}")
                
                with col2:
                    st.subheader("🎤 Notable Artists")
                    for artist in genre_info['artists']:
                        st.write(f"🌟 {artist}")
                    
                    st.subheader("🎸 Key Instruments")
                    for instrument in genre_info['key_instruments']:
                        st.write(f"🎼 {instrument}")
            
            else:
                st.error("❌ Failed to analyze the audio file. Please try another file.")
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)

def dataset_tab():
    """Dataset information and analysis tab."""
    st.header("📊 Dataset Analysis")
    
    # Dataset overview
    st.subheader("📁 Available Datasets")
    
    datasets = {
        "30-second features": "data/features_30_sec.csv",
        "3-second features": "data/features_3_sec.csv"
    }
    
    for name, path in datasets.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(f"{name} - Samples", len(df))
            with col2:
                st.metric("Features", len(df.columns) - 1)
            with col3:
                st.metric("Genres", df['label'].nunique())
            with col4:
                st.metric("File Size", f"{os.path.getsize(path) / 1024 / 1024:.1f} MB")
            
            # Genre distribution
            genre_counts = df['label'].value_counts()
            fig = px.bar(
                x=genre_counts.index,
                y=genre_counts.values,
                title=f"Genre Distribution - {name}",
                labels={'x': 'Genre', 'y': 'Number of Samples'}
            )
            fig.update_layout(template='plotly_white')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(f"Dataset not found: {path}")
    
    # Genre information
    st.subheader("🎵 Supported Genres")
    
    genre_cols = st.columns(5)
    for i, genre in enumerate(GENRES):
        with genre_cols[i % 5]:
            with st.container():
                st.markdown(f"### {genre.title()}")
                info = get_genre_info(genre)
                st.write(f"*{info['origin']}*")
                st.write(f"Tempo: {info['tempo']}")

def about_tab():
    """About and project information tab."""
    st.header("ℹ️ About Music Genre Classifier Pro")
    
    st.markdown("""
    ### 🎵 Project Overview
    This is an advanced Music Genre Classification system that uses deep learning to automatically 
    identify the genre of audio files. The system analyzes various audio features and uses a 
    neural network to classify music into 10 different genres.
    
    ### 🧠 How It Works
    1. **Feature Extraction**: The system extracts 58 different audio features including:
       - MFCCs (Mel-frequency cepstral coefficients)
       - Chroma features
       - Spectral contrast
       - Zero crossing rate
       - Spectral centroid and rolloff
    
    2. **Deep Learning Model**: A 6-layer neural network with ~767K parameters
       - Dense layers with ReLU activation
       - Dropout for regularization
       - Softmax output for classification
    
    3. **Training Process**: The model is trained on a dataset of audio features
       - Configurable epochs and early stopping
       - Real-time training monitoring
       - Comprehensive evaluation metrics
    
    ### 📊 Features
    - **Audio Prediction**: Upload audio files for genre classification
    - **Model Training**: Train new models with custom parameters
    - **Data Analysis**: Explore dataset statistics and distributions
    - **Audio Visualization**: Waveform and spectrogram analysis
    - **Performance Metrics**: Detailed model evaluation and reports
    
    ### 🎯 Supported Genres
    """)
    
    genre_info_text = ", ".join([genre.title() for genre in GENRES])
    st.write(f"**{genre_info_text}**")
    
    st.markdown("""
    ### 🔧 Technical Specifications
    - **Framework**: TensorFlow/Keras
    - **Audio Processing**: Librosa
    - **Web Interface**: Streamlit
    - **Visualization**: Plotly
    - **Supported Audio Formats**: WAV, MP3, FLAC, M4A
    - **Maximum File Size**: 50MB
    - **Sample Rate**: 22,050 Hz
    - **Analysis Duration**: 30 seconds
    
    ### 📈 Model Performance
    The current model achieves approximately 74% accuracy on the test set, with performance 
    varying by genre. Some genres like Classical and Jazz tend to have higher accuracy due 
    to their distinctive musical characteristics.
    
    ### 🚀 Getting Started
    1. **For Prediction**: Go to the Prediction tab and upload an audio file
    2. **For Training**: Use the Training tab to train a new model
    3. **For Analysis**: Explore the Dataset tab for data insights
    
    ### 💡 Tips for Best Results
    - Use high-quality audio files
    - Ensure files are at least 3 seconds long
    - Classical and jazz music typically classify more accurately
    - Pop and rock genres might have more overlap
    """)

def main():
    # Header
    st.markdown('<h1 class="main-header">🎵 Music Genre Classifier Pro</h1>', unsafe_allow_html=True)
    st.markdown("### Advanced AI-powered music genre classification with training capabilities")
    
    # Sidebar navigation
    with st.sidebar:
        st.header("🎛️ Navigation")
        
        # Tab selection
        tabs = {
            "🎯 Prediction": "prediction",
            "🏋️ Training": "training", 
            "📊 Dataset": "dataset",
            "ℹ️ About": "about"
        }
        
        selected_tab = st.radio("Select a tab:", list(tabs.keys()), index=0)
        
        if selected_tab is None:
            selected_tab = list(tabs.keys())[0]
        
        current_tab = tabs[selected_tab]
        
        st.markdown("---")
        
        # Model information
        st.header("📊 System Info")
        st.info(f"""
        **Supported Genres:** {len(GENRES)} genres
        
        **Audio Formats:** {', '.join(SUPPORTED_FORMATS)}
        
        **Max File Size:** {MAX_FILE_SIZE}MB
        
        **Sample Rate:** {SAMPLE_RATE} Hz
        
        **Features:** 58 audio features
        """)
        
        # Quick stats
        if current_tab == "prediction":
            predictor = load_predictor()
            if predictor:
                st.success("✅ Model loaded")
            else:
                st.warning("⚠️ No model found")
        
        elif current_tab == "training":
            if st.session_state.training_in_progress:
                st.warning("⏳ Training in progress")
            elif st.session_state.training_results:
                results = st.session_state.training_results
                if 'error' not in results:
                    st.success(f"✅ Last training: {results['test_accuracy']:.1%} accuracy")
        
        st.markdown("---")
        st.markdown("**🎵 Music Genre Classifier Pro**")
        st.markdown("*Powered by Deep Learning*")
    
    # Main content based on selected tab
    if current_tab == "prediction":
        prediction_tab()
    elif current_tab == "training":
        training_tab()
    elif current_tab == "dataset":
        dataset_tab()
    elif current_tab == "about":
        about_tab()

if __name__ == "__main__":
    main()
