import os

import librosa
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.models import load_model


@tf.keras.utils.register_keras_serializable()
class Attention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1), initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1), initializer='zeros', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        return K.sum(x * a, axis=1)

    def get_config(self):
        return super(Attention, self).get_config()


SR = 16000
FRAME_MS = 0.025
HOP_MS = 0.010
N_FFT = int(SR * FRAME_MS)
HOP_LENGTH = int(SR * HOP_MS)
SEQ_LEN = 300
CLASSES = ['Angry', 'Sad', 'Neutral', 'Happy']


def extract_inference_features(audio_path):
    audio, _ = librosa.load(audio_path, sr=SR)
    audio = audio / (np.max(np.abs(audio)) + 1e-8)

    mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=128)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=40)

    stft = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH))
    contrast = librosa.feature.spectral_contrast(S=stft, sr=SR)

    chroma = librosa.feature.chroma_stft(S=stft, sr=SR)

    combined = np.vstack((log_mel, mfcc, contrast, chroma))
    mean = np.mean(combined, axis=1, keepdims=True)
    std = np.std(combined, axis=1, keepdims=True) + 1e-8
    combined = (combined - mean) / std
    features = combined.T  # (Time, 187)

    if features.shape[0] < SEQ_LEN:
        pad_width = SEQ_LEN - features.shape[0]
        features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
    else:
        features = features[:SEQ_LEN, :]

    return np.expand_dims(features, axis=0)


def predict_emotion(audio_file_path, model_path="models/emotion_model_final_v2.keras"):
    model = load_model(model_path, custom_objects={'Attention': Attention})

    input_data = extract_inference_features(audio_file_path)

    probs = model.predict(input_data, verbose=0)[0]
    pred_idx = np.argmax(probs)

    print(f"\nResult for: {os.path.basename(audio_file_path)}")
    print("-" * 30)
    for i, label in enumerate(CLASSES):
        print(f"{label:10}: {probs[i] * 100:.2f}%")
    print("-" * 30)
    print(f"FINAL PREDICTION: {CLASSES[pred_idx]}")
