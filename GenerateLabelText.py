import csv
import argparse
import os 

def load_transcripts(csv_file):
    transcripts = {}
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            audio_file = row[0]
            transcript = row[1]
            transcripts[audio_file] = transcript
    return transcripts

def create_transcript_file(tsv_file, csv_file, output_file):
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    transcripts = load_transcripts(csv_file)

    with open(tsv_file, 'r') as f, open(output_file, 'w') as out:
        reader = csv.reader(f, delimiter='\t')
        next(reader)  # Skip the header row
        for row in reader:
            audio_file = row[0].split(' ')[0]
            audio_file = audio_file.split('/')[-1]
            audio_file = audio_file.replace('.wav', '')
            transcript = transcripts.get(audio_file)
            if transcript is not None:
                out.write(transcript + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsv_file', required=True, help='The tsv file to read from.')
    parser.add_argument('--csv_file', required=True, help='The csv file to find transcripts in.')
    parser.add_argument('--output_file', required=True, help='The output text file.')
    args = parser.parse_args()

    create_transcript_file(args.tsv_file, args.csv_file, args.output_file)