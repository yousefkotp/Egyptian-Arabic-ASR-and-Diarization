import argparse
import random
import pandas as pd
import numpy as np
from pathlib import Path
import secrets
from openai import OpenAI
from pydub import AudioSegment
import os

# Parse command line arguments
parser = argparse.ArgumentParser(description="Generate speech from transcripts in a CSV file and update outputs incrementally.")
parser.add_argument("--csv_file", type=str, help="Path to the CSV file containing transcripts.")
parser.add_argument("--output_dir", type=str, help="Path to the output directory where the audio files will be saved.")
parser.add_argument("--sample_rate", type=int, help="Output sample rate for the audio files.", default=16000)
args = parser.parse_args()

# Initialize OpenAI client
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Ensure output directory exists
output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Path to the mapping CSV file
mapping_csv_path = output_dir / "synthetic.csv"

# Check if mapping CSV exists, if not create it
if not mapping_csv_path.exists():
    pd.DataFrame(columns=["audio", "transcript"]).to_csv(mapping_csv_path, index=False)

# Read the CSV file
df = pd.read_csv(args.csv_file)

# Check if 'transcript' column exists
if 'transcript' not in df.columns:
    raise ValueError("CSV file does not contain a 'transcript' column.")

# List of available voices
voices = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]

for index, row in df.iterrows():
    # Randomly select a voice
    selected_voice = random.choice(voices)
    
    # Uniformly sample a speed between 0.8 and 1.2
    speed = np.random.uniform(0.8, 1.2)
    
    # Generate a random 16-character filename
    filename = secrets.token_hex(8)  # Generates 16 characters
    
    # Define the path for the generated speech file
    speech_file_path = output_dir / f"{filename}.wav"
    
    # Create speech from transcript
    response = client.audio.speech.create(
      model="tts-1",
      voice=selected_voice,
      input=row['transcript'],
      response_format="wav",
      speed=speed
    )
    
    # Save the generated audio to file
    response.write_to_file(speech_file_path)

    # Convert the sample rate to the specified rate using pydub
    if args.sample_rate != 24000:
        audio = AudioSegment.from_wav(speech_file_path)
        audio = audio.set_frame_rate(args.sample_rate)
        audio.export(speech_file_path, format="wav")
    
    # Append the new mapping to the synthetic.csv file
    with open(mapping_csv_path, 'a', newline='', encoding='utf-8') as f:
        pd.DataFrame([{"audio": filename, "transcript": row['transcript']}]).to_csv(f, header=False, index=False)

    # Remove the processed transcript from the original CSV
    df.drop(index, inplace=True)
    df.to_csv(args.csv_file, index=False)

    print(f"Processed and saved: {filename}")

print("All transcripts have been processed.")