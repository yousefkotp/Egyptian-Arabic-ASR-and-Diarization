def create_vocab_file(spm_vocab_path, output_vocab_path):
    with open(spm_vocab_path, 'r', encoding='utf-8') as spm_file:
        with open(output_vocab_path, 'w', encoding='utf-8') as out_file:
            for line in spm_file:
                # Each line in the .vocab file is 'token\t<score>\n'
                token = line.split('\t')[0]
                out_file.write(token + '\n')

# Set the paths for the input .vocab file from SentencePiece and the output vocab.txt
spm_vocab_path = 'tokenizer.vocab'
output_vocab_path = 'vocab.txt'

# Generate vocab.txt
create_vocab_file(spm_vocab_path, output_vocab_path)
