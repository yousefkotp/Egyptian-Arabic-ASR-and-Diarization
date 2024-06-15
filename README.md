# ASR_for_egyptian_dialect

## ArTST (Text to Speech)

### Normalize
```bash
python .\AraTSTnormalization.py --input_file=data/train.csv --output_file=data/train_artst_normalized.csv
python .\AraTSTnormalization.py --input_file=data/adapt.csv --output_file=data/adapt_artst_normalized.csv
```

### Generate Speaker Embeddings
```bash
python .\SpeakerEmbedding.py --input_dir=data/train --output_dir=data/speaker_embedding/train
python .\SpeakerEmbedding.py --input_dir=data/adapt --output_dir=data/speaker_embedding/adapt
```

### Execlude Foreign Characters
```bash
python .\ExecludeForeignChars.py --input_file=data/train_artst_normalized.csv --output_file=data/train_artst_normalized_filtered.csv --char_file=allowed_chars.txt  
python .\ExecludeForeignChars.py --input_file=data/adapt_artst_normalized.csv --output_file=data/adapt_artst_normalized_filtered.csv --char_file=allowed_chars.txt  
```

### Generate Manifest File
```bash
python .\GenerateManifest.py --audio_dir=data/train --embedding_dir=data/speaker_embedding/train --output_file=data/train.tsv --audio_csv=data/train_artst_normalized_filtered.csv
python .\GenerateManifest.py --audio_dir=data/adapt --embedding_dir=data/speaker_embedding/adapt --output_file=data/adapt.tsv --audio_csv=data/adapt_artst_normalized_filtered.csv
```

### Generate Labels
```bash
python GenerateLabelText.py --tsv_file=data/train.tsv --csv_file=data/train_artst_normalized_filtered.csv --output_file=data/train_labels.txt
python GenerateLabelText.py --tsv_file=data/adapt.tsv --csv_file=data/adapt_artst_normalized_filtered.csv --output_file=data/labels/adapt.txt
```