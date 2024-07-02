"""
    This script checks the sampling frequency of .wav files in the given directories.
    It prints a message for each file that has a sampling frequency different than 16KHz.

    Example usage:
        python CheckSamplingFrequency.py data/train data/adapt
"""
import wave
import os
import argparse

def check_sampling_frequency(directory):
    """
    This function checks the sampling frequency of .wav files in the given directory.
    
    Args:
        directory (str): The directory to check.
    """
    print(f"Processing directory: {directory}")
    for filename in os.listdir(directory):
        if filename.endswith(".wav"):
            filepath = os.path.join(directory, filename)
            print(f"Checking file: {filename}")
            with wave.open(filepath, 'rb') as wav_file:
                if wav_file.getframerate() != 16000:
                    print(f"The file {filename} has a sampling frequency different than 16KHz.")
    print(f"Finished processing directory: {directory}")

if __name__ == "__main__":
    """
    This script checks the sampling frequency of .wav files in the given directories.
    It prints a message for each file that has a sampling frequency different than 16KHz.
    """
    parser = argparse.ArgumentParser(description='Check the sampling frequency of .wav files in the given directories.')
    parser.add_argument('directories', type=str, nargs='+',
                        help='a list of directories to check')
    args = parser.parse_args()

    for directory in args.directories:
        check_sampling_frequency(directory)