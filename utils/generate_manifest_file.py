"""
Example usage:
python .\GenerateManifest.py --audio_dir=data/train --embedding_dir=data/speaker_embedding/train --output_file=data/train.tsv --audio_csv=data/train_artst_normalized_filtered.csv

"""
import os
import csv
import argparse
import wave
import pandas as pd

def create_manifest(audio_dir, audio_csv, embedding_dir, output_file):
    """
    Creates a manifest file for wav2vec2.0.

    Args:
        audio_dir (str): The directory containing audio files.
        audio_csv (str): The CSV file containing audio file names.
        embedding_dir (str): The directory containing embedding files.
        output_file (str): The path to the output tsv file.
    """
    # Check if the output directory exists and create it if it doesn't
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(['/'])

        df = pd.read_csv(audio_csv)
        audio_files = df['audio'].tolist()
        embedding_files = [f for f in os.listdir(embedding_dir) if f.endswith('.npy')]
        for index, embedding_file in enumerate(embedding_files):
            if index % 5000 == 0:
                print(f'Processing {index} out of {len(embedding_files)} files.')
            
            audio_file = embedding_file.replace('.npy', '')
            audio_file = audio_file.replace('.wav', '')
            if not audio_file in audio_files:
                print(f'Audio file {audio_file} not found in CSV file. Skipping...')
                continue
            
            audio_file = audio_file + '.wav'
            audio_path = os.path.join(audio_dir, audio_file)
            with wave.open(audio_path, 'rb') as wav_file:
                n_samples = wav_file.getnframes()
            writer.writerow([os.path.abspath(audio_path), n_samples, os.path.abspath(os.path.join(embedding_dir, embedding_file))])
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create a manifest file in the form acceptable by wav2vec2.0.')
    parser.add_argument('--audio_dir', type=str, required=True, help='The directory containing audio files.')
    parser.add_argument('--audio_csv', type=str, required=True, help='The CSV file containing audio file names.')
    parser.add_argument('--embedding_dir', type=str, required=True, help='The directory containing embedding files.')
    parser.add_argument('--output_file', type=str, required=True, help='The path to the output tsv file.')
    args = parser.parse_args()

    create_manifest(args.audio_dir, args.audio_csv, args.embedding_dir, args.output_file)