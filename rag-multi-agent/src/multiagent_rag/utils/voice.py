import os
import re
import speech_recognition as sr


class VoiceHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer()

        # Keep your sensitivity settings
        self.recognizer.energy_threshold = 150
        self.recognizer.dynamic_energy_threshold = False

    def listen(self) -> str:
        try:
            # KEEP your device_index=10 (or whichever one worked for your headset!)
            with sr.Microphone(device_index=2) as source:
                print("🎙️ Listening... (Speak loudly into your mic)")
                audio = self.recognizer.listen(source, timeout=8, phrase_time_limit=15)

            print("⏳ Sound detected! Transcribing...")

            # ⚡ FIX 1: Use Google's Free STT (Filters out static MUCH better than Whisper)
            text = self.recognizer.recognize_google(audio)

            # ⚡ FIX 2: The "Pinecone Crash" Protection
            # Remove all weird punctuation hallucinations (like ". . . .")
            clean_text = re.sub(r'[^a-zA-Z0-9\s]', '', text).strip()

            # If the result is empty after cleaning, it was just noise. Ignore it.
            if not clean_text:
                print("⚠️ Ignored static/background noise.")
                return ""

            print(f"✅ Transcribed: '{clean_text}'")
            return clean_text

        except sr.UnknownValueError:
            # Google throws this when it hears audio but it's just mumbling/static
            print("⚠️ Could not understand the audio. (Just static?)")
            return ""
        except sr.WaitTimeoutError:
            print("⚠️ No speech detected. (Timed out)")
            return ""
        except Exception as e:
            print(f"❌ Voice Error: {e}")
            return ""