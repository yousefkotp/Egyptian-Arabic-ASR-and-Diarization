"""
Example usage:
python .\ExecludeForeignChars.py --input_file=data/adapt_artst_normalized.csv --output_file=data/adapt_artst_normalized_filtered.csv --char_file=allowed_chars.txt  
"""
import pandas as pd
import argparse

def filter_transcript(transcript, valid_chars):
    return ''.join(c for c in transcript if c in valid_chars)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",'--input_file', required=True, help='Input CSV file with text')
    parser.add_argument("-o",'--output_file', required=True, help='Output CSV file for filtered text')
    parser.add_argument("-c",'--char_file', required=True, help='File with valid characters')
    args = parser.parse_args()

    # Load valid characters from file
    with open(args.char_file, 'r', encoding='utf-8') as f:
        valid_chars = set(line.strip() for line in f)
    valid_chars.add(' ')  # Add space to the set of valid characters
    print(f'Loaded {len(valid_chars)} valid characters.')
    print(f'Valid characters: {valid_chars}')
    df = pd.read_csv(args.input_file)
    for i in df.index:
        df.at[i, 'transcript'] = filter_transcript(df.at[i, 'transcript'], valid_chars)
    df.to_csv(args.output_file, index=False)