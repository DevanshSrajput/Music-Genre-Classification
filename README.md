# 🎸 Music Genre Classifier Pro 🎶

_Because who needs Shazam when you have this?_

---

## 📸 Screenshots
> Glimpses! So you know it actually works

<div align="center">
<table>
<tr>
<td align="center"><img src="Screenshots/Dashboard.png" width="300"><br><b>🏠 Home Dashboard</b></td>
<td align="center"><img src="Screenshots/Training.png" width="300"><br><b>📤 Training Tab</b></td>
<td align="center"><img src="Screenshots/Trained Results.png" width="300"><br><b>📤 Training Results</b></td>
</tr>
<tr>
<td align="center"><img src="Screenshots/Trained Results-2.png" width="300"><br><b>📊 Trained Results</b></td>
<td align="center"><img src="Screenshots/Trained Result-3.png" width="300"><br><b>💬 Trained Results</b></td>
<td align="center"><img src="Screenshots/Dataset.png" width="300"><br><b>📤 Dataset</b></td>
</tr>
<tr>
<td align="center"><img src="Screenshots/Dataset-2.png" width="300"><br><b>🤖 Dataset</b></td>
<td align="center"><img src="Screenshots/Prediction.png" width="300"><br><b>📊 Prediction</b></td>
<td align="center"><img src="Screenshots/Predict-2.png" width="300"><br><b>📤 Predicted Results</b></td>
</tr>
<tr>
<td align="center"><img src="Screenshots/Predict-3.png" width="300"><br><b>⚙️ Predicted Results</b></td>
<td align="center"><img src="Screenshots/About.png" width="300"><br><b>🌙 About</b></td>
</tr>
</table>
</div>

---

## 🤔 What is This?

Welcome to **Music Genre Classifier Pro** – the AI-powered app that bravely attempts to guess your music genre, even if you upload your grandma’s humming or a 10-hour loop of elevator music. Built for musicians, data nerds, and anyone who’s ever wondered, “Is this really jazz?”

---

## 🚀 Features

- 🎤 **Predicts music genre** from audio files (wav, mp3, flac, ogg)
- 🧠 **Deep learning model** trained on 10 genres (yes, including “pop” for your guilty pleasures)
- 📊 **Training & dataset analysis** right in the app
- 🖼️ **Beautiful Streamlit UI** (because looks matter)
- 🛠️ **Custom feature extraction** (58 features, because 57 wasn’t enough)
- 💾 **No unnecessary files** (unless you count your music taste)

---

## 🛠️ Tech Stack

- **Python 3.8+** (because Python 2 is so last decade)
- **TensorFlow / Keras** (for the neural net magic)
- **Librosa** (audio wrangling)
- **Streamlit** (for that slick web UI)
- **Scikit-learn, Pandas, Matplotlib, Seaborn** (data science essentials)
- **Pydub** (audio file wizardry)

---

## 🗂️ Project Structure

```
Music-Genre-Classification/
│
├── app_enhanced.py         # Streamlit web app
├── train_model.py          # Model training script
├── predict.py              # CLI prediction script
├── utils/
│   ├── audio_processor.py  # Audio feature extraction
│   └── data_loader.py      # Data loading & preprocessing
├── models/
│   └── genre_classifier.py # The neural network model
├── data/                   # Datasets (CSV/features/audio)
├── results/                # Training results & plots
├── config.py               # All the magic numbers
├── requirements.txt        # All the dependencies
└── README.md               # This masterpiece
```

---

## 📦 Prerequisites

- **Python 3.8 or higher** (recommended: 3.10+)
- **pip** (Python package manager)
- **Git** (for cloning the repository)
- **FFmpeg** (for MP3/FLAC/OGG support in audio processing)
  - Download from: https://ffmpeg.org/download.html
  - Make sure ffmpeg is in your system PATH
- **A modern web browser** (for the Streamlit UI)

---

## 🏃 How to Run

1. **Clone this repo**  
   ```
   git clone https://github.com/DevanshSrajput/music-genre-classification.git`
   ```

2. **Install dependencies**  
   ```
   pip install -r requirements.txt`
   ```

3. **(Optional) Train the model**  
   ```
   python train_model.py`
   ```

4. **Launch the app**  
   ```
   streamlit run app_enhanced.py
   ```

5. **Quick Start**
   ```
   python run quick_start.py
   ```

6. **Upload your audio file**

   - Supported: `.wav`, `.mp3`, `.flac`, `.ogg`
   - Max size: 10MB (because we care about your bandwidth)

7. **Marvel at the predictions**
   - Or question your life choices if it says “Pop” for your death metal track.


---

## 📬 Contact

**Devansh Singh**

- Email: dksdevansh@gmail.com
- GitHub: [DevanshSrajput](https://github.com/DevanshSrajput)
- If you find a bug, blame the AI. If you love it, star the repo!

---

## ⚠️ Disclaimer

This project is for educational purposes. If it mislabels your favorite blues track as pop, consider it a feature, not a bug.  
Enjoy, and may your genres always be
