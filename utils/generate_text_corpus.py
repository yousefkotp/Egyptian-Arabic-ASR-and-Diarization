import pandas as pd
import numpy as np
from openai import OpenAI
import os

client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

base_prompt = """
Generate a list of unique sentences in Egyptian Arabic that reflect the local culture, everyday life, and diverse topics. Each sentence should be distinct, capturing various aspects of life in Egypt, including local expressions and cultural references. The sentences should be suitable for text-to-speech applications, focusing on clarity and naturalness in the Egyptian dialect.

Your output should include a mix of sentence types, such as statements, questions, exclamations, and commands. Aim for a diverse set of sentences that showcase the richness and vibrancy of the Egyptian Arabic language. Consider incorporating common phrases, idioms, and colloquial expressions to add authenticity to the sentences.

Your output should be in the following format:
- Each sentence should start on a new line.
- Avoid repetitive content and ensure that each sentence is unique.
- Use correct spelling, grammar, and punctuation to maintain readability.
- Aim for a balanced mix of sentence lengths and structures to create engaging and varied content.
- Output format should be only and only one sentence per line totally in Arabic without any other words or formatting.
- Generate as much as possible senstences per request/prompt.
- Try to be as egyptian as possible
- You can go into politics, religion, culture, history, or any other topic that is relevant to Egypt.
- The average number of words per sentence should be around 13 sentences, it is totally fine if it is less or more.
- Avoid using `-` in the beginning of the sentence.
"""

unique_sentences = set()
target_sentences = 30000
output_file = 'output_sentences.txt'
prompt_counter = 1

# Open the file in write mode
with open(output_file, 'w', encoding='utf-8') as file:
    while len(unique_sentences) < target_sentences:
        dynamic_prompt = f"{base_prompt}\nThis is request number {prompt_counter}."

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": dynamic_prompt}
            ],
            max_tokens=4096,
            temperature=0.7,
            n=1,
        )

        sentences = response.choices[0].message.content.split('\n')

        for sentence in sentences:
            # Ensure the sntence is at least two words long and not already in the set
            if len(sentence.split()) >= 2 and sentence not in unique_sentences:
                print(sentence)
                unique_sentences.add(sentence)
                file.write(sentence + '\n')

        prompt_counter += 1

        print(f"Progress: {len(unique_sentences)}/{target_sentences} sentences generated.")

    print("Generation complete. Sentences saved to output_sentences.txt.")