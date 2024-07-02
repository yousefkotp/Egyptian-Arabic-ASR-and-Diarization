def count_words_in_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
        words = content.split()
        return len(words)

if __name__ == "__main__":
    file_path = input("Enter the file path: ")
    word_count = count_words_in_file(file_path)
    print(f"The file contains {word_count} words.")