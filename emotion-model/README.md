# Speech Emotion Recognition System (Deep Learning)

## Overview

This project implements a Speech Emotion Recognition (SER) system using deep learning. It processes raw audio signals, extracts advanced acoustic features, and predicts human emotions from speech.

The model classifies audio into four categories:

* Angry
* Sad
* Neutral
* Happy

The system combines signal processing techniques with neural networks (CNN, BiLSTM, and Attention) to achieve reliable performance.

---

## Project Structure

```
├── data_preprocessing.py      # Feature extraction and dataset preparation
├── train_model.py             # Model training pipeline
├── inference.py               # Emotion prediction on new audio
├── Processed_Data_Seq/        # Saved NumPy datasets
├── model/                     # Trained model (.keras)
├── results/                   # Outputs, predictions, plots
└── README.md                  # Project documentation
```

---

## Key Features

### Audio Feature Extraction

* Log Mel Spectrogram (128)
* MFCC (40)
* Spectral Contrast
* Chroma Features
* Per-feature normalization

### Sequence Chunking

* Audio is split into overlapping sequences
* Sequence length: 300 frames (~3 seconds)
* Step size: 150 (50% overlap)

### Model Architecture

* Conv1D layers for local temporal patterns
* Batch Normalization and Dropout for regularization
* Bidirectional LSTM for temporal dependencies
* Attention layer to focus on important segments
* Dense layers for classification

---

## Model Architecture

```
Input (300, 187)
   ↓
Conv1D (128) + BatchNorm + Dropout + MaxPool
   ↓
Conv1D (256) + BatchNorm + Dropout + MaxPool
   ↓
Bidirectional LSTM (128)
   ↓
Attention Layer
   ↓
Dense (64)
   ↓
Output (4 classes - Softmax)
```

---

## Training Details

* Optimizer: Adam (learning rate = 0.001)
* Loss: Sparse Categorical Crossentropy
* Batch size: 32
* Epochs: Up to 30

### Callbacks

* ModelCheckpoint (saves best model)
* ReduceLROnPlateau
* EarlyStopping

---

## Dataset Processing

### Pipeline

1. Load `.parquet` dataset containing audio bytes
2. Convert bytes to waveform
3. Normalize audio
4. Extract features
5. Segment into sequences
6. Map original labels to 4 classes
7. Save as NumPy arrays

---

## Emotion Mapping

| Original Emotion | Mapped  |
| ---------------- | ------- |
| angry            | Angry   |
| sad              | Sad     |
| neutral          | Neutral |
| happy/excited    | Happy   |
| others           | Removed |

---

## Evaluation Metrics

* Accuracy
* Confusion Matrix
* Classification Report (Precision, Recall, F1-score)

---

## Inference Pipeline

### Steps

1. Load trained model
2. Load audio file
3. Extract features
4. Generate chunks
5. Predict each chunk
6. Average probabilities
7. Return final prediction

### Example Output

```
Predicted Emotion: Happy (92.34%)
Chunks analyzed: 5

Probabilities:
- Angry: 1.12%
- Sad: 2.45%
- Neutral: 4.09%
- Happy: 92.34%
```

---

## How to Run

### Install Dependencies

```bash
pip install numpy pandas librosa soundfile tensorflow scikit-learn matplotlib seaborn tqdm
```

### Train Model

```bash
python train_model.py
```

### Run Inference

```bash
python inference.py
```

---

## Model File

```
emotion_model_final_v2.keras
```

---

## Outputs Generated

* final_prediction_values.csv
* Confusion matrix visualization
* Training accuracy and loss plots

---

## Future Improvements

* Extend to more emotion classes
* Experiment with transformer-based models
* Real-time audio prediction
* Deploy as a web application (Flask or FastAPI)
* Mobile deployment

---

## Contributing

Pull requests are welcome. For major changes, open an issue first.

---

## Acknowledgements

* Librosa for audio processing
* TensorFlow/Keras for deep learning
* Scikit-learn for evaluation tools

---

## Author

Thevinu Dassanayake
