import os
import re
import tempfile

import speech_recognition as sr

from multiagent_rag.utils.logger import get_logger

logger = get_logger(__name__)


class STTEngine:

    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 300
        self.recognizer.dynamic_energy_threshold = False

    def transcribe(self, audio_path: str) -> str:
        try:
            logger.info(f"Starting transcription for: {audio_path}")

            ext = os.path.splitext(audio_path)[1].lower()

            if ext in [".wav", ".flac", ".aiff"]:
                audio_file_path = audio_path
            else:
                audio_file_path = self._convert_to_wav(audio_path)

            with sr.AudioFile(audio_file_path) as source:
                audio_data = self.recognizer.record(source)

            text = self.recognizer.recognize_google(audio_data)
            clean_text = re.sub(r'[^a-zA-Z0-9\s\?\.\,\!\']', '', text).strip()

            if not clean_text:
                logger.warning("Transcription returned empty text after cleaning")
                return ""

            logger.info(f"Transcription successful: '{clean_text[:100]}...'")
            return clean_text

        except sr.UnknownValueError:
            logger.warning("Could not understand the audio content")
            return ""
        except sr.RequestError as e:
            logger.error(f"STT service request failed: {str(e)}")
            return ""
        except Exception as e:
            logger.error(f"Transcription failed: {str(e)}")
            return ""

    def _convert_to_wav(self, input_path: str) -> str:
        try:
            from pydub import AudioSegment

            audio = AudioSegment.from_file(input_path)
            wav_path = tempfile.mktemp(suffix=".wav")
            audio.export(wav_path, format="wav")
            logger.info(f"Converted {input_path} to WAV format")
            return wav_path
        except ImportError:
            logger.warning("pydub not installed, trying direct file read")
            return input_path
        except Exception as e:
            logger.error(f"Audio conversion failed: {str(e)}")
            return input_path
