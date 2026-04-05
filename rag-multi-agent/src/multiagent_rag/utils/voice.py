import os
import re
import speech_recognition as sr


class VoiceHandler:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 150
        self.recognizer.dynamic_energy_threshold = False

    def listen(self) -> str:
        try:
            with sr.Microphone(device_index=2) as source:
                audio = self.recognizer.listen(source, timeout=8, phrase_time_limit=15)

            text = self.recognizer.recognize_google(audio)
            clean_text = re.sub(r"[^a-zA-Z0-9\s]", "", text).strip()

            if not clean_text:
                return ""

            return clean_text

        except sr.UnknownValueError:
            return ""
        except sr.WaitTimeoutError:
            return ""
        except Exception:
            return ""