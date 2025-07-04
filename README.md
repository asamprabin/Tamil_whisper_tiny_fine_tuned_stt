# 🗣️ Tamil Whisper STT: Efficient Speech-to-Text for Tamil

Welcome to **Tamil Whisper STT** – an open-source, production-ready pipeline for high-quality Tamil speech-to-text (STT) using state-of-the-art Whisper models. This project is designed for fast, accurate, and resource-efficient transcription, making it ideal for deployment on small devices and scalable systems alike.

---

## 🚀 Features

- **Fine-tuned Whisper Tiny Model**: Custom-trained on the Indic Tamil dataset for optimal Tamil ASR performance.
- **End-to-End Pipeline**: From data preprocessing to model training and fast inference.
- **Multi-Device Support**: Seamless GPU/CPU compatibility.
- **Fast Inference**: Leverage [faster-whisper](https://github.com/SYSTRAN/faster-whisper) for blazing-fast transcription.
- **Open & Reproducible**: All scripts, notebooks, and utilities included for full transparency and reproducibility.

---

## 📂 Project Structure

```
dataset/
├── wavs_original/         # Raw audio files (.wav)
├── wavs_converted/        # Audio files converted to 16kHz mono
├── metadata.csv           # Training metadata (filtered, CSV)
├── metadata.txt           # Raw metadata (TXT)
convert_audio_sample_rate.sh  # Audio conversion script
metadata_pre.py               # Metadata preprocessing script
fine_tune_whisper.ipynb       # Model fine-tuning notebook
fine_tuned_inference.py       # Inference with fine-tuned model
original_inference.py         # Inference with base Whisper model
fast_inference.py             # Fast inference using faster-whisper
requirements.txt              # Python dependencies
test.wav                      # Example audio file
readme.md                     # Project documentation
```

---

## 🏁 Quickstart

### 1️⃣ Prepare Metadata

Ensure your metadata is in the correct format and transcriptions are less than 447 tokens.

```txt
train_tamilfem_05396    அதுக்கு அப்புறமா அவனோட புகழ் உலகம் முழுசும் பரவும்னு சொன்னாரு .
```

- **Convert and filter metadata:**
  ```sh
  python metadata_pre.py
  ```
  See `metadata_pre.py` for details.

### 2️⃣ Convert Audio

Convert all audio files to 16kHz mono for Whisper compatibility.

```sh
bash convert_audio_sample_rate.sh
```
See `convert_audio_sample_rate.sh`.

### 3️⃣ Fine-Tune Whisper Model

Train the Whisper Tiny model on your dataset.

- Open and run `fine_tune_whisper.ipynb` for step-by-step fine-tuning.
- Uses HuggingFace Transformers and Datasets.

### 4️⃣ Inference

#### a. **Fine-Tuned Model**
```sh
python fine_tuned_inference.py
```
- See `fine_tuned_inference.py`.

#### b. **Original Whisper Model**
```sh
python original_inference.py
```
- See `original_inference.py`.

#### c. **Fast Inference (faster-whisper)**
```sh
python fast_inference.py
```
- See `fast_inference.py`.

---

## 🧑‍💻 Technical Highlights

- **Data Preprocessing**: `metadata_pre.py` ensures only valid-length transcriptions are used.
- **Audio Conversion**: `convert_audio_sample_rate.sh` standardizes audio for training/inference.
- **Model Training**: `fine_tune_whisper.ipynb` leverages HuggingFace's Trainer API with a custom data collator for efficient batching.
- **Inference**: Multiple scripts for benchmarking and deployment:
  - `fine_tuned_inference.py`: Inference with your fine-tuned model.
  - `original_inference.py`: Baseline inference with the original Whisper Tiny.
  - `fast_inference.py`: Ultra-fast inference using [faster-whisper](https://github.com/SYSTRAN/faster-whisper).

---

## 📦 Requirements

- Python 3.10+
- [transformers](https://github.com/huggingface/transformers)
- [datasets](https://github.com/huggingface/datasets)
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- sox (for audio conversion)

Install dependencies:
```sh
pip install -r requirements.txt
```

---

## 📝 Citation

If you use this project, please consider citing or starring the repository!

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Please open an issue or submit a pull request.


---

## 📬 Contact

For questions or collaborations, please open an issue or reach out via GitHub.

---

**Empowering Tamil language technology, one open-source model

