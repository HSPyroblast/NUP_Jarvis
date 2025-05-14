import os
import time
import queue
import subprocess
import contextlib
import pvporcupine
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
with open(os.devnull, 'w') as fnull:
    with contextlib.redirect_stdout(fnull):
        import pygame

from gtts import gTTS
from logger import logger
from dotenv import load_dotenv


# --- ENV SETUP ---
load_dotenv()
picovoice_access_key = os.getenv('PICOVOICE_API_KEY')


# --- CONSTANTS ---
SAMPLE_RATE = 16000
CHANNELS = 1
SILENCE_THRESHOLD = 0.5
SILENCE_DURATION = 2
WHISPER_CPP_PATH = '../whisper.cpp/build/bin/whisper-cli.exe'
WHISPER_MODEL_PATH = '../whisper.cpp/models/ggml-base.bin'
AUDIO_DIR = '../data/audio'

os.makedirs(AUDIO_DIR, exist_ok=True)


# --- AUDIO RECORDING ---
q_audio = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    q_audio.put(indata.copy())

def record_audio_to_wav(filename=f'{AUDIO_DIR}/recording.wav'):
    frames = []
    silence_start = None
    logger.info("Recording (auto-stop on silence)...")

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=audio_callback):
        while True:
            data = q_audio.get()
            frames.append(data)

            volume = np.linalg.norm(data)
            if volume < SILENCE_THRESHOLD:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_DURATION:
                    break
            else:
                silence_start = None

    audio = np.concatenate(frames, axis=0)
    wav.write(filename, SAMPLE_RATE, audio)
    logger.info(f'Saved: {filename}')
    return filename


# --- WHISPER TRANSCRIPTION ---
def transcribe_with_whisper_cpp(audio_path: str):
    logger.info('Recognizing via whisper.cpp...')
    base_path = audio_path.replace(".wav", "")
    txt_path = base_path + ".txt"

    result = subprocess.run([
        WHISPER_CPP_PATH,
        '-m', WHISPER_MODEL_PATH,
        '-f', audio_path,
        '--language', 'ru',
        '-of', base_path,
        '--output-txt',
        '--no-speech-thold', '0.5'
    ], capture_output=True, text=True)

    if result.returncode != 0:
        logger.info(f'Error: {result.stderr}')
        return ''

    if os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""


# --- GOOGLE TTS + PLAYBACK ---
def generate_tts(text: str, lang='en'):
    logger.info(f"TTS: '{text}'")
    filename = f"{AUDIO_DIR}/tts.mp3"

    tts = gTTS(text=text, lang=lang, slow=False)
    tts.save(filename)
    return filename

def play_tts(filename: str):
    logger.info("Audio playback...")
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    pygame.mixer.music.stop()
    pygame.mixer.quit()

# --- WAKE WORD LISTENER ---
porcupine = None
q_porcupine = queue.Queue()

def sd_callback(indata, frames, time_info, status):
    pcm = (indata[:, 0] * 32768).astype(np.int16)
    q_porcupine.put(pcm)

def listen_for_activation(wake_word="americano"):
    global porcupine
    porcupine = pvporcupine.create(keywords=[wake_word], access_key=picovoice_access_key)

    with sd.InputStream(
            samplerate=porcupine.sample_rate,
            channels=1,
            dtype='float32',
            blocksize=porcupine.frame_length,
            callback=sd_callback
    ):
        try:
            logger.info("Waiting for wake word...")
            while True:
                pcm = q_porcupine.get()
                keyword_index = porcupine.process(pcm)
                if keyword_index >= 0:
                    logger.info(f"Wake word '{wake_word}' detected!")
                    return 1
        except KeyboardInterrupt:
            logger.info("Interrupted by the user.")
        finally:
            porcupine.delete()
    return 0


# --- TEST ---
if __name__ == "__main__":
    file = generate_tts('Похоже все работает!', 'ru')
    play_tts(file)