import csv

def append_transcripts_to_file(csv_file_path, text_file_path):
    with open(csv_file_path, mode='r', newline='', encoding='utf-8') as csv_file:
        reader = csv.DictReader(csv_file)
        with open(text_file_path, mode='a', encoding='utf-8') as text_file:
            for row in reader:
                transcript = row['transcript']
                text_file.write(transcript + '\n')

if __name__ == "__main__":
    csv_file_path = input("Enter the path to the CSV file: ")
    text_file_path = input("Enter the path to the text file where transcripts will be appended: ")
    append_transcripts_to_file(csv_file_path, text_file_path)
    print("Transcripts have been appended to the text file.")