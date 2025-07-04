from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration
import time

# Path to your fine-tuned model directory
model_dir = "./whisper-tamil-finetuned/checkpoint-6750"  # change if different

# Load processor from base model (tokenizer files) and feature extractor from fine-tuned model
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained(model_dir)

# Create pipeline
asr = pipeline(
    task="automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    device=0,  # use 0 if running on GPU, or remove for CPU
    # generate_kwargs={"language": "<|ta|>", "task": "transcribe"}
)

# Inference
start_time = time.time()
result = asr("test_denoise.wav")  # path to your test audio
end_time = time.time()

with open("output.txt", "w", encoding="utf-8") as f:
    f.write(result["text"])

print(f"Inference time: {end_time - start_time:.2f} seconds")