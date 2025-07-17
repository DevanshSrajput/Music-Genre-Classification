"""Quick start script for Music Genre Classification project."""

import os
import sys
import subprocess

def check_file_exists(filepath, description):
    """Check if a file exists and provide feedback."""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ {description}: {filepath} (NOT FOUND)")
        return False

def check_dataset():
    """Check if dataset files are available."""
    print("🔍 Checking dataset files...")
    
    dataset_files = [
        ("data/features_30_sec.csv", "30-second features CSV"),
        ("data/features_3_sec.csv", "3-second features CSV"),
        ("features_30_sec.csv", "30-second features CSV (root)"),
        ("features_3_sec.csv", "3-second features CSV (root)"),
        ("data/genres_original/", "Original audio files folder"),
        ("genres_original/", "Original audio files folder (root)")
    ]
    
    found_files = []
    for filepath, description in dataset_files:
        if check_file_exists(filepath, description):
            found_files.append(filepath)
    
    if not found_files:
        print("\n❌ No dataset files found!")
        print("Please ensure you have one of the following:")
        print("1. features_30_sec.csv (recommended)")
        print("2. features_3_sec.csv")
        print("3. genres_original/ folder with audio files")
        return False
    
    return True

def install_requirements():
    """Install required packages."""
    print("\n📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def train_model():
    """Train the model."""
    print("\n🚀 Starting model training...")
    try:
        subprocess.check_call([sys.executable, "train_model.py"])
        print("✅ Model training completed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during training: {e}")
        return False

def launch_app():
    """Launch the Streamlit app."""
    print("\n🌐 Launching Streamlit app...")
    try:
        subprocess.check_call([sys.executable, "-m", "streamlit", "run", "app_enhanced"
        ".py"])
    except subprocess.CalledProcessError as e:
        print(f"❌ Error launching app: {e}")
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")

def main():
    """Main function."""
    print("🎵 Music Genre Classification - Quick Start")
    print("=" * 50)
    
    # Check if dataset is available
    if not check_dataset():
        return
    
    # Check if model exists
    model_exists = check_file_exists("models/genre_classifier.h5", "Trained model")
    
    if not model_exists:
        print("\n🤔 No trained model found. Let's train one!")
        
        # Ask user if they want to install requirements
        install_req = input("\n📦 Install/update requirements? (y/n, default=y): ").strip().lower()
        if install_req != 'n':
            if not install_requirements():
                return
        
        # Train the model
        train_choice = input("\n🚀 Start training? (y/n, default=y): ").strip().lower()
        if train_choice != 'n':
            if not train_model():
                return
        else:
            print("❌ Cannot proceed without a trained model.")
            return
    
    # Launch the app
    print("\n🎉 Everything is ready!")
    launch_choice = input("🌐 Launch Streamlit app? (y/n, default=y): ").strip().lower()
    if launch_choice != 'n':
        launch_app()
    else:
        print("👍 You can manually launch the app with: streamlit run app.py")
        print("📱 Or make predictions with: python predict.py <audio_file>")

if __name__ == "__main__":
    main()