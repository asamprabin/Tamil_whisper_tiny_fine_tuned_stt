mkdir -p dataset/wavs_converted

# Convert all .wav files in the original dataset to 16kHz mono
# Ensure you have sox installed: sudo apt-get install sox libsox-fmt-all
# This script assumes the original files are in dataset/wavs_original/
# and will save the converted files in dataset/wavs_converted/
for file in dataset/wavs_original/*.wav; do
    sox "$file" -r 16000 -c 1 "dataset/wavs_converted/$(basename "$file")"
done