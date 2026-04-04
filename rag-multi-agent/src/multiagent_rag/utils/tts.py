import os
import asyncio
import edge_tts

# Hide the pygame welcome message from cluttering the terminal
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame


class TTSEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TTSEngine, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        print("[TTS] Initializing Microsoft Edge Neural Engine...")
        pygame.mixer.init()

    # We use an async wrapper function to handle the edge-tts generation safely
    async def _generate_audio(self, text: str, voice: str, audio_file: str):
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(audio_file)

    def speak(self, text: str):
        if not text:
            return

        try:
            print("🔊 Generating voice...")
            audio_file = "temp_response.mp3"

            # Using the professional female voice we just tested
            voice_model = "en-US-AriaNeural"

            # Generate and save the audio
            asyncio.run(self._generate_audio(text, voice_model, audio_file))

            # Play the audio
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()

            # Wait for it to finish playing before moving on
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)

            # Unload and clean up the file
            pygame.mixer.music.unload()
            if os.path.exists(audio_file):
                os.remove(audio_file)

        except Exception as e:
            print(f"❌ TTS Error: {e}")