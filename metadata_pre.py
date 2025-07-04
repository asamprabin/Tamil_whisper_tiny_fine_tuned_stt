from transformers import WhisperTokenizer

# Load tokenizer (adjust to your model if needed)
tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-tiny")


input_file = "dataset/metadata.txt"
output_file = "dataset/metadata.csv"


# Convert metadata from txt to csv format
# Ensure the transcription is less than 447 tokens
# and save it in a CSV file with columns: file, text
# The input file is expected to be in the format: file_id \t transcription
with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    fout.write("file,text\n")
    for line in fin:
        parts = line.rstrip('\n').split('\t', 1)
        if len(parts) == 2:
            file_id, transcription = parts
            token_ids = tokenizer(transcription).input_ids
            if len(token_ids) < 447:
                fout.write(f"{file_id},\"{transcription}\"\n")
        else:
            fout.write(line)
