"""
SpeakerEmbedding.py

This script gets the speaker embeddings for .wav files in a given directory and saves them to an output directory.
It uses the SpeechBrain library to compute the embeddings.

Example usage:
    python .\SpeakerEmbedding.py --input_dir=data/adapt --output_dir=data/speaker_embedding/adapt
"""
# pip install git+https://github.com/speechbrain/speechbrain.git@develop
import os
import argparse
import torchaudio
import numpy as np
from tqdm import tqdm
from speechbrain.inference.speaker import EncoderClassifier

def get_speaker_embeddings(input_dir, output_dir):
    """
    This function gets the speaker embeddings for .wav files in the given directory and saves them to the output directory.
    
    Args:
        input_dir (str): The directory containing .wav files.
        output_dir (str): The directory to save the speaker embeddings.
    """
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb")
    files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
    for filename in tqdm(files, desc="Processing files"):
        filepath = os.path.join(input_dir, filename)
        signal, fs = torchaudio.load(filepath)
        embeddings = classifier.encode_batch(signal)
        embeddings = embeddings.squeeze() 
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        np.save(os.path.join(output_dir, filename + '.npy'), embeddings.detach().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get speaker embeddings for .wav files in the given directory and save them to the output directory.')
    parser.add_argument('--input_dir', type=str, help='The directory containing .wav files.')
    parser.add_argument('--output_dir', type=str, help='The directory to save the speaker embeddings.')
    args = parser.parse_args()
    os.system('pip install git+https://github.com/speechbrain/speechbrain.git@develop')

    get_speaker_embeddings(args.input_dir, args.output_dir)