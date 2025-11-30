
# Speech Recognition Using Deep Learning

A Convolutional Neural Network (CNN) based speaker recognition system that identifies speakers from audio files using MFCC (Mel-Frequency Cepstral Coefficients) features.

## 📋 Overview

This project implements a deep learning model for speaker recognition using:
- **Feature Extraction**: MFCC features extracted from audio files
- **Model Architecture**: CNN with convolutional and pooling layers
- **Classification**: Multi-class classification using softmax activation

## 🏗️ Architecture Pipeline

```
Audio file (.wav)
    ↓
Feature extraction (MFCCs / Spectrogram)
    ↓
Deep Learning Model (CNN or LSTM)
    ↓
Softmax Layer → predicts speaker ID
```

## 🔧 Model Architecture

The CNN model consists of:
- **Input Layer**: 40 x 200 MFCC features
- **Conv2D Layer 1**: 32 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D Layer 1**: 2x2 pool size
- **Conv2D Layer 2**: 64 filters, 3x3 kernel, ReLU activation
- **MaxPooling2D Layer 2**: 2x2 pool size
- **Flatten Layer**: Converts 2D features to 1D
- **Dense Layer**: 128 neurons, ReLU activation
- **Dropout Layer**: 0.3 dropout rate
- **Output Layer**: Softmax activation for speaker classification

**Total Parameters**: 3,165,575 (12.08 MB)

## 📊 Performance

- **Training Accuracy**: ~98.46% (Epoch 5)
- **Validation Accuracy**: ~98.67% (Epoch 5)
- **Test Accuracy**: **98.67%**

### Training History

| Epoch | Training Acc | Training Loss | Val Acc | Val Loss |
|-------|-------------|---------------|---------|----------|
| 1     | 69.95%      | 1.3769        | 98.67%  | 0.0589   |
| 2     | 97.18%      | 0.1016        | 98.80%  | 0.0379   |
| 3     | 99.08%      | 0.0386        | 98.80%  | 0.0298   |
| 4     | 98.97%      | 0.0352        | 98.40%  | 0.0491   |
| 5     | 98.46%      | 0.0381        | 98.67%  | 0.0438   |

## 🚀 Getting Started

### Prerequisites

```bash
pip install librosa numpy soundfile scikit-learn tensorflow
```

### Installation

1. Mount Google Drive (if using Google Colab):
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. Extract the dataset:
```python
import zipfile
import os

zip_path = '/content/drive/MyDrive/dataset_speech/archive_dataset.zip'
extract_path = '/content/drive/MyDrive/dataset_speech/'

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)
```

### Dataset Structure

```
dataset_speech/
└── 16000_pcm_speeches/
    ├── Speaker_1/
    │   ├── 0.wav
    │   ├── 1.wav
    │   └── ...
    ├── Speaker_2/
    │   ├── 0.wav
    │   └── ...
    └── ...
```

**Dataset Statistics**: 7,507 audio samples

## 💻 Usage

### Feature Extraction

```python
import librosa
import numpy as np

def extract_features(file_path, max_pad_len=200):
    audio, sample_rate = librosa.load(file_path, sr=None, mono=True)
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    
    # Padding/Truncating to fixed length
    pad_width = max_pad_len - mfccs.shape[1]
    if pad_width > 0:
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    
    return mfccs
```

### Training the Model

```python
history = model.fit(
    X_train, y_train,
    epochs=5,
    batch_size=32,
    validation_data=(X_test, y_test)
)
```

### Making Predictions

```python
test_file = "path/to/audio/file.wav"
feature = extract_features(test_file)

if feature is not None:
    prediction = model.predict(feature[np.newaxis, ..., np.newaxis])
    predicted_speaker = le.inverse_transform([np.argmax(prediction)])
    print("Recognized Speaker:", predicted_speaker[0])
```

## 📁 Project Structure

```
project/
├── README.md
├── dataset/
│   └── 16000_pcm_speeches/
├── models/
│   └── speaker_recognition_model.h5
└── notebooks/
    └── speech_recognition.ipynb
```

## 🔍 Key Features

- **Robust Audio Processing**: Handles both standard WAV files and raw PCM format
- **Fallback Mechanism**: Uses FFmpeg for problematic audio files
- **Fixed-Length Features**: MFCC features padded/truncated to 200 frames
- **Data Augmentation Ready**: Architecture supports various audio preprocessing techniques
- **High Accuracy**: Achieves 98.67% test accuracy with minimal epochs

## 🛠️ Technologies Used

- **Python 3.12**
- **TensorFlow/Keras**: Deep learning framework
- **Librosa**: Audio processing and feature extraction
- **NumPy**: Numerical computations
- **Scikit-learn**: Data preprocessing and evaluation
- **FFmpeg**: Audio format conversion (fallback)

## 📈 Future Improvements

- [ ] Implement LSTM/GRU layers for temporal feature learning
- [ ] Add data augmentation (pitch shifting, time stretching)
- [ ] Experiment with mel-spectrograms and raw waveforms
- [ ] Deploy as a REST API
- [ ] Add real-time speaker recognition
- [ ] Implement transfer learning with pre-trained models

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👥 Authors

- Your Name - Initial work

## 🙏 Acknowledgments

- Dataset source: [Specify dataset source]
- Librosa library for audio processing
- TensorFlow/Keras team for the deep learning framework

---

**Note**: This model was trained on Google Colab with the dataset stored in Google Drive. Adjust paths accordingly for your environment.
```
