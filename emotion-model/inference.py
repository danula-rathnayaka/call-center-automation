import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K


SR = 16000
FRAME_MS = 0.025
HOP_MS = 0.010
N_FFT = int(SR * FRAME_MS)
HOP_LENGTH = int(SR * HOP_MS)

SEQ_LEN = 300
STEP = 150    
CLASSES = ['Angry', 'Sad', 'Neutral', 'Happy']



@tf.keras.utils.register_keras_serializable()
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

    def get_config(self):
        return super(Attention, self).get_config()


def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio

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


class EmotionPredictor:
    def __init__(self, model_path):
        """Loads the model upon initialization to avoid reloading it for every audio file."""
        print(f"Loading model from {model_path}...")
        self.model = load_model(model_path, custom_objects={'Attention': Attention})
        print("✅ Model loaded successfully!")

    def predict(self, audio_path):
        """Processes an audio file and returns the predicted emotion."""
       
        try:
            audio, _ = librosa.load(audio_path, sr=SR, mono=True)
        except Exception as e:
            return f"Error loading audio: {e}"
        
 
        audio = normalize_audio(audio)
        
  
        features = extract_premium_features(audio, SR)
        num_frames = features.shape[0]
        
        chunks = []
        
 
        if num_frames < SEQ_LEN:
         
            pad_width = SEQ_LEN - num_frames
            padded_features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant')
            chunks.append(padded_features)
        else:
          
            for start in range(0, num_frames - SEQ_LEN + 1, STEP):
                end = start + SEQ_LEN
                chunks.append(features[start:end, :])
            
       
            if num_frames % STEP != 0 and num_frames > SEQ_LEN:
                chunks.append(features[-SEQ_LEN:, :])
                
        chunks = np.array(chunks, dtype=np.float32)
        
   
        predictions = self.model.predict(chunks, verbose=0)
        

        avg_probabilities = np.mean(predictions, axis=0)
        final_prediction_idx = np.argmax(avg_probabilities)
        final_emotion = CLASSES[final_prediction_idx]
        
 
        result = {
            "Final_Emotion": final_emotion,
            "Confidence": f"{avg_probabilities[final_prediction_idx] * 100:.2f}%",
            "Chunk_Count": len(chunks),
            "All_Probabilities": {CLASSES[i]: f"{avg_probabilities[i] * 100:.2f}%" for i in range(len(CLASSES))}
        }
        
        return result


if __name__ == "__main__":
   
    MODEL_PATH = "/kaggle/input/datasets/thevinu/ghcgfy6/emotion_model_final_v2 (1).keras"  
    AUDIO_PATH = "/kaggle/input/datasets/thevinu/sd23r2df23/Vocaroo 02 Apr 2026 13_29_44 GMT0530 1mcnT1AvABgh.wav" 
    
   
    if os.path.exists(MODEL_PATH):
        predictor = EmotionPredictor(MODEL_PATH)
        
      
        if os.path.exists(AUDIO_PATH):
            print(f"\nAnalyzing: {AUDIO_PATH}")
            results = predictor.predict(AUDIO_PATH)
            
            print("\n--- Prediction Results ---")
            print(f"Predicted Emotion: {results['Final_Emotion']} ({results['Confidence']})")
            print(f"Number of 3-second chunks analyzed: {results['Chunk_Count']}")
            print("Detailed Probabilities:")
            for emotion, prob in results['All_Probabilities'].items():
                print(f"  - {emotion}: {prob}")
        else:
            print(f"❌ Audio file not found at: {AUDIO_PATH}")
    else:
        print(f"❌ Model file not found at: {MODEL_PATH}")