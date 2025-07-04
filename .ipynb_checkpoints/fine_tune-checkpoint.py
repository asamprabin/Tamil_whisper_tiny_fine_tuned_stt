from datasets import Dataset, Audio
import pandas as pd
from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration, TrainingArguments, Trainer

df = pd.read_csv("dataset/metadata.csv")
df["file"] = df["file"].apply(lambda x: f"dataset/wavs_converted/{x}.wav")
dataset = Dataset.from_pandas(df)
dataset = dataset.cast_column("file", Audio(sampling_rate=16000))

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language="ta", task="transcribe")

def prepare(example):
    audio = example["file"]
    inputs = processor(audio["array"], sampling_rate=16000, return_tensors="pt")
    input_values = inputs.input_features.squeeze(0)
    labels = processor.tokenizer(example["text"], return_tensors="pt").input_ids.squeeze(0)
    return {"input_features": input_values, "labels": labels}

processed_dataset = dataset.map(prepare)

print("Processed dataset:")

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")

training_args = TrainingArguments(
    output_dir="./whisper-tamil-finetuned",
    per_device_train_batch_size=8,
    evaluation_strategy="no",
    learning_rate=1e-5,
    num_train_epochs=10,
    save_steps=500,
    fp16=True,
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_dataset,
    tokenizer=processor.feature_extractor,
)

trainer.train()