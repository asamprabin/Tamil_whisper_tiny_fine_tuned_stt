from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
import time
import torch

# Use the base Whisper Tiny model
model_name = "openai/whisper-tiny"

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load processor and model
processor = WhisperProcessor.from_pretrained(model_name)
model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device, dtype=torch.float16)

# Create pipeline with language set to Tamil ("ta")
asr = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=0,  # use 0 for GPU, or remove this line for CPU
    # generate_kwargs={"language": "<|ta|>", "task": "transcribe"}
)

# Inference
start_time = time.time()
result = asr("test.wav")
end_time = time.time()

# Save result
with open("output_original.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])

print(f"Inference time: {end_time - start_time:.2f} seconds")
