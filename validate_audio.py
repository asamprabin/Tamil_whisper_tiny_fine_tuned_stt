import os
import wave

folder = "dataset/wavs_converted"

# Validate audio files in the specified folder
for filename in os.listdir(folder):
    if filename.endswith(".wav"):
        path = os.path.join(folder, filename)
        with wave.open(path, "rb") as wav_file:
            channels = wav_file.getnchannels()
            framerate = wav_file.getframerate()
            print(f"{filename}: {framerate} Hz, {channels} channel(s)")