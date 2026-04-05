import os, io, warnings
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Dense, LSTM, Dropout, BatchNormalization, 
                                     Conv1D, MaxPooling1D, Bidirectional, Layer, 
                                     SpatialDropout1D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

PARQUET_FOLDER = "/kaggle/input/datasets/thevinu/dataset" 
SAVE_FOLDER = "/kaggle/working/Processed_Data_Seq"
os.makedirs(SAVE_FOLDER, exist_ok=True)

X_PATH = os.path.join(SAVE_FOLDER, "X_seq.npy")
Y_PATH = os.path.join(SAVE_FOLDER, "Y_seq.npy")

SR = 16000
FRAME_MS = 0.025
HOP_MS = 0.010
N_FFT = int(SR * FRAME_MS)
HOP_LENGTH = int(SR * HOP_MS)

SEQ_LEN = 300 
STEP = 150     

EMOTION_COLS = ['frustrated', 'angry', 'sad', 'disgust', 'excited',
                'fear', 'neutral', 'surprise', 'happy']

def extract_premium_features(audio, sr):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=128)
    log_mel = librosa.power_to_db(mel, ref=np.max)
    
    mfcc = librosa.feature.mfcc(S=log_mel, n_mfcc=40)
    
    stft = np.abs(librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH))
    contrast = librosa.feature.spectral_contrast(S=stft, sr=sr)
    
    chroma = librosa.feature.chroma_stft(S=stft, sr=sr)
    
    combined = np.vstack((log_mel, mfcc, contrast, chroma))
    
    mean = np.mean(combined, axis=1, keepdims=True)
    std = np.std(combined, axis=1, keepdims=True) + 1e-8
    combined = (combined - mean) / std
    
    return combined.T

def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio

def audio_from_bytes(audio_dict):
    audio_bytes = audio_dict['bytes'] if isinstance(audio_dict, dict) else audio_dict
    audio_array, sr = sf.read(io.BytesIO(audio_bytes))
    
    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
        
    if sr != SR:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=SR)
    return audio_array

def process_row(row):
    try:
        audio = audio_from_bytes(row['audio'])
        audio = normalize_audio(audio)

        features = extract_premium_features(audio, SR)
        file_label = row[EMOTION_COLS].to_numpy(dtype=np.float32)

        num_frames = features.shape[0]
        chunks_X = []
        chunks_Y = []

        for start in range(0, num_frames - SEQ_LEN + 1, STEP):
            end = start + SEQ_LEN
            chunk = features[start:end, :]
            
            chunks_X.append(chunk)
            chunks_Y.append(file_label)

        return chunks_X, chunks_Y
    except Exception as e:
        return [], []

X_all = []
Y_all = []

print("🚀 Starting Sequence Preprocessing on Kaggle...")

if not os.path.exists(PARQUET_FOLDER):
    print(f"❌ Error: Dataset path not found: {PARQUET_FOLDER}")
    print("Please check your Kaggle Input paths.")
else:
    parquet_files = [f for f in os.listdir(PARQUET_FOLDER) if f.lower().endswith(".parquet")]
    
    for pq_file in parquet_files:
        pq_path = os.path.join(PARQUET_FOLDER, pq_file)
        df = pd.read_parquet(pq_path)
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {pq_file}"):
            new_X_chunks, new_Y_chunks = process_row(row)

            if len(new_X_chunks) > 0:
                X_all.extend(new_X_chunks)
                Y_all.extend(new_Y_chunks)

    if len(X_all) == 0:
        print("❌ Error: No data processed. Check if audio files are shorter than 3 seconds.")
    else:
        X_final = np.array(X_all, dtype=np.float32)
        Y_final = np.array(Y_all, dtype=np.float32)

        print("\n✅ Finished Extraction!")
        print(f"X Shape: {X_final.shape} (Chunks, Time, Features)")
        print(f"Y Shape: {Y_final.shape} (Chunks, Classes)")

        np.save(X_PATH, X_final)
        np.save(Y_PATH, Y_final)
        print(f"💾 Saved successfully to {SAVE_FOLDER}")


DATA_FOLDER = "/kaggle/working/Processed_Data_Seq"

print("🚀 Loading raw sequence data...")
X_raw = np.load(os.path.join(DATA_FOLDER, "X_seq.npy"))
Y_raw = np.load(os.path.join(DATA_FOLDER, "Y_seq.npy"))

print(f"Original Count: {len(X_raw)} chunks")

mapping_array = np.array([
    -1,  
     0,  
     1,  
    -1,  
     3,  
    -1,  
     2,  
    -1,  
     3   
])

print("⚡ Filtering and Merging via NumPy Vectorization...")

old_labels = np.argmax(Y_raw, axis=1)

new_labels = mapping_array[old_labels]

valid_indices = new_labels != -1

X_final = X_raw[valid_indices]
Y_final = new_labels[valid_indices].astype(np.int64)

print("-" * 30)
print(f"✅ Cleanup Complete!")
print(f"Original Chunks: {len(X_raw)}")
print(f"Final Chunks:    {len(X_final)}")
print("-" * 30)

unique, counts = np.unique(Y_final, return_counts=True)
classes = ['Angry', 'Sad', 'Neutral', 'Happy']

print("\n📊 New Class Distribution:")
for u, c in zip(unique, counts):
    print(f"{classes[u]}: {c} chunks")

np.save(os.path.join(DATA_FOLDER, "X_final.npy"), X_final)
np.save(os.path.join(DATA_FOLDER, "Y_final.npy"), Y_final)

print(f"\n💾 Saved 'X_final.npy' and 'Y_final.npy' to {DATA_FOLDER}")


MODEL_SAVE_PATH = "/kaggle/working/emotion_model_final_v2.keras"
X_PATH = os.path.join(DATA_FOLDER, "X_final.npy")
Y_PATH = os.path.join(DATA_FOLDER, "Y_final.npy")

print("🚀 Loading Premium Data...")
X = np.load(X_PATH)
Y = np.load(Y_PATH)

X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y
)


class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        return K.sum(x * a, axis=1)

inputs = Input(shape=(300, 187))

x = Conv1D(128, kernel_size=5, padding='same', activation='relu')(inputs)
x = BatchNormalization()(x)
x = SpatialDropout1D(0.45)(x)
x = MaxPooling1D(pool_size=2)(x)

x = Conv1D(256, kernel_size=3, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = SpatialDropout1D(0.45)(x)
x = MaxPooling1D(pool_size=2)(x)

x = Bidirectional(LSTM(128, return_sequences=True))(x)
x = Dropout(0.5)(x)

x = Attention()(x)

x = Dense(64, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
outputs = Dense(4, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

checkpoint = ModelCheckpoint(MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001, verbose=1)
early_stop = EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)

print("\n🔥 Training with Improved Regularization...")
history = model.fit(
    X_train, y_train, validation_data=(X_test, y_test), 
    epochs=30, batch_size=32, 
    callbacks=[checkpoint, lr_reducer, early_stop]
)

print("\n" + "="*50)
print("📊 FINAL PREDICTION REPORT (ALL CLASS VALUES)")
print("="*50)

model.load_weights(MODEL_SAVE_PATH)
probs = model.predict(X_test)
preds = np.argmax(probs, axis=1)

results = pd.DataFrame(probs, columns=[f"{c}_Prob" for c in CLASSES])
results['True_Label'] = [CLASSES[i] for i in y_test]
results['Predicted_Label'] = [CLASSES[i] for i in preds]

cols = ['True_Label', 'Predicted_Label'] + [f"{c}_Prob" for c in CLASSES]
results = results[cols]

print("\n--- Sample of Predictions (First 15 Chunks) ---")
print(results.head(15).to_string(index=False))

results.to_csv("/kaggle/working/final_prediction_values.csv", index=False)
print(f"\n✅ All {len(results)} test samples with their 4-class values saved to 'final_prediction_values.csv'")

print("\n---Classification_Report  ---")
print(classification_report(y_test, preds, target_names=CLASSES))

cm = confusion_matrix(y_test, preds)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix Heatmap")
plt.show()

plt.figure(figsize=(10,4))
plt.plot(history.history['accuracy'], label='Train Acc', marker='o')
plt.plot(history.history['val_accuracy'], label='Val Acc', marker='o')
plt.title("Training & Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10,4))
plt.plot(history.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
plt.title("Training & Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()