"""
AraTSTnormalization.py

This script is used for normalizing Arabic text. It performs the following operations:
- Converts Arabic numerals to their corresponding digits
- Removes diacritics from the text
- Removes punctuation from the text with the exception of @ and %.
- Drops rows with no transcript

The script takes as input a CSV file with text and outputs a CSV file with the normalized text.
"""

import re
import sys
import argparse
import unicodedata
import os 
import pyarabic.araby as araby
import pandas as pd

map_numbers = {'0': '٠', '1': '١', '2': '٢', '3': '٣', '4': '٤', '5': '٥', '6': '٦', '7': '٧', '8': '٨', '9': '٩'}
map_numbers = dict((v, k) for k, v in map_numbers.items())
punctuations = ''.join([chr(i) for i in list(i for i in range(sys.maxunicode) if unicodedata.category(chr(i)).startswith('P'))])
punctuations = punctuations + '÷#ݣ+=|$×⁄<>`åûݘ ڢ̇ پ'

def convert_numerals_to_digit(word):
    """
    Converts Arabic numerals in the word to their corresponding digits.
    """
    sentence=[]
    for w in word:
        sentence.append(map_numbers.get(w, w))
    word = ''.join(sentence)
    return word

def remove_diacritics(word):
    """
    Removes diacritics from the word.
    """
    return araby.strip_diacritics(word)

def remove_punctuation(word):
    """
    Removes punctuation from the word.
    """
    return word.translate(str.maketrans('', '', re.sub('[@% ]','', punctuations))).lower()

def normalize_text(sentence, index):
    """
    Normalizes the sentence by converting numerals to digits, removing diacritics, and removing punctuation.
    If the sentence is not a string, it prints out the index and corresponding audio name and returns None.
    """
    if not isinstance(sentence, str):
        print(f"Non-string value at index {index}: {sentence}")
        print(f"Corresponding audio name: {df.at[index, 'audio']}")
        return None
    sentence = convert_numerals_to_digit(sentence)
    sentence = remove_diacritics(sentence)
    sentence = remove_punctuation(sentence)
    return sentence

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i",'--input_file', required=True, help='Input CSV file with text')
    parser.add_argument("-o",'--output_file', required=True, help='Output CSV file for normalized text')
    args = parser.parse_args()
    
    df = pd.read_csv(args.input_file)
    for i in df.index:
        df.at[i, 'transcript'] = normalize_text(df.at[i, 'transcript'], i)
    df = df.dropna(subset=['transcript'])
    df.to_csv(args.output_file, index=False)