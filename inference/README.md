# Inference for Speaker Diarization with ASR for Egyptian Dialect

This repository contains scripts and models for automatic speech recognition (ASR) and speaker diarization, tailored for the Egyptian dialect.
Follow the instructions below to clone the repository, install dependencies, and run inference.

## Cloning the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/AbdelrhmanElnenaey/ASR_for_egyptian_dialect
cd ASR_for_egyptian_dialect
```


## Installation Guide

### System Dependencies

Install necessary system packages.

```bash
apt-get install -y sox libsndfile1 ffmpeg
```

### Python Dependencies

Install Python packages using pip. Ensure you have pip installed before running this command.

```bash
pip install -r requirements.txt
```

For NeMo, install the specific branch required.

```bash
BRANCH='main'
pip install git+https://github.com/NVIDIA/NeMo.git@$BRANCH#egg=nemo_toolkit[asr]
```

## Running the Scripts
### 1. Modify Diarizer

Run `modify_diarizer.py` to apply modifications to NeMo's `clustering_diarizer.py` script:

```bash

python modify_diarizer.py
```

### 2. Generate Manifest

Generate a manifest file using `generate_manifest.py`. This file is required for the diarization inference process.

```bash
python generate_manifest.py \
    --paths2audio_files audio_files.txt \
    --data_dir /path/to/audio_files \
    --manifest_filepath test_manifest.json \
    --add_duration
```
Arguments:

  `--paths2audio_files`: Path to a text file containing the list of audio file paths (This is generated internally, no need to modify).
  
  `--data_dir`: Directory where the audio files are located.
  
  `--manifest_filepath`: Path to save the generated manifest file.
  
  `--add_duration`: Optional flag to add duration information to the manifest.

### 3. Inference

Run `inference.py` to perform ASR and diarization.

```bash
python inference.py \
    --asr_model /path/to/asr_model.ckpt \
    --data_dir /path/to/audio_files \
    --asr_output results.csv \
    --input_manifest_path test_manifest.json \
    --output_manifest_path test_manifest_vocals.json \
    --temp_output_dir temp_outputs \
    --mono_output_dir temp_outputs_mono \
    --config_path /path/to/config.yaml
```
Arguments:

  `--asr_model`: Path to the pre-trained ASR model checkpoint.
  
  `--data_dir`: Directory where the audio files are located.
  
  `--asr_output`: Path to save ASR results.
  
  `--input_manifest_path`: Path to the input manifest file.
  
  `--output_manifest_path`: Path to save the output manifest file after preprocessing.
  
  `--temp_output_dir`: Directory to store temporary output files.
  
  `--mono_output_dir`: Directory to store mono channel output files (final dir that contains wav files after preprocessing).
  
  `--config_path`: Path to the configuration YAML file. This is located in `configs/FC-transducer-inference.yaml`.
  

### Additional Notes

  If running on Kaggle or Google Colab, you might need to restart the runtime after running `modify_diarizer.py`:

``` python
import os
os._exit(00)
```
An inference notebook was also added for reference.

Ensure to adjust file paths and directory names as per your setup.

