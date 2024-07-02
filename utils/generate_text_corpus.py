import pandas as pd
import numpy as np
from openai import OpenAI
import os

# Initialize OpenAI client
client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

# Base part of the prompt for generating sentences in Egyptian Arabic
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

# Initialize a set to store unique sentences
unique_sentences = set()

# Target number of unique sentences
target_sentences = 100000

# File to save the sentences
output_file = 'output_sentences.txt'

# Counter for dynamic prompt modification
prompt_counter = 1

# Open the file in write mode
with open(output_file, 'w', encoding='utf-8') as file:
    # Continue generating sentences until the target is reached
    while len(unique_sentences) < target_sentences:
        # Modify the prompt with a dynamic component
        dynamic_prompt = f"{base_prompt}\nThis is request number {prompt_counter}."

        # Generate text using the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": dynamic_prompt}
            ],
            max_tokens=4096,  # Adjust as needed
            temperature=0.7,  # Adjust for creativity
            n=1,  # Number of completions
        )

        # Split the response into sentences
        sentences = response.choices[0].message.content.split('\n')

        # Filter and add unique sentences to the set
        for sentence in sentences:
            # Ensure the sentence is at least two words long and not already in the set
            if len(sentence.split()) >= 2 and sentence not in unique_sentences:
                print(sentence)
                unique_sentences.add(sentence)
                # Write the unique sentence to the file
                file.write(sentence + '\n')

        # Increment the prompt counter for the next iteration
        prompt_counter += 1

        # Optional: Print progress
        print(f"Progress: {len(unique_sentences)}/{target_sentences} sentences generated.")

    print("Generation complete. Sentences saved to output_sentences.txt.")