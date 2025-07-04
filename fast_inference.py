from faster_whisper import WhisperModel
import time

# Explicitly set to GPU if available
model = WhisperModel("small", compute_type="float16", device="cuda")

# Inference
start_time = time.time()
segments, _ = model.transcribe("test.wav", beam_size=5, task="transcribe", language="ta")
end_time = time.time()

# Save results to a file
with open("output_fast.txt", "w", encoding="utf-8") as f:
    for segment in segments:
        f.write(segment.text + "\n")

# Print inference time
print(f"Inference time: {end_time - start_time:.2f} seconds")