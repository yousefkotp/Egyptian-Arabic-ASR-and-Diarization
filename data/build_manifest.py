import librosa
import json
from tqdm import tqdm
import os

def build_manifest(data_path, output_path, filename, split, take=-1):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    file_path = os.path.join(output_path, filename)
    with open(file_path, "w+") as fout:
        with open(f"{data_path}/{split}.csv", "r") as fp:
            header = True
            for line in tqdm(fp):
                if header:
                    header = False
                    continue

                line = line.strip()
                data = line.split(",")
                sample_path = f"{data_path}/{split}/{data[0]}.wav"
                try:
                    duration = librosa.get_duration(path=sample_path, sr=16000)
                    sample = {
                        "audio_filepath": sample_path,
                        "duration": duration,
                        "text": data[1]
                    }
                    json.dump(sample, fout, ensure_ascii = False)
                    fout.write("\n")
                    if take > 0:
                        take -= 1
                    if take == 0:
                        break
                except:
                    continue

build_manifest("data", "manifest_files", "train_manifest.json", "train")
build_manifest("data", "manifest_files", "adapt_manifest.json", "adapt")
build_manifest("data", "manifest_files", "synthetic_manifest.json", "synthetic")