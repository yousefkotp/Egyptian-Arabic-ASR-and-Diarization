import sentencepiece as spm

spm.SentencePieceTrainer.train('--input=out.txt --model_prefix=tokenizer --vocab_size=32000 --character_coverage=1 --model_type=bpe')
