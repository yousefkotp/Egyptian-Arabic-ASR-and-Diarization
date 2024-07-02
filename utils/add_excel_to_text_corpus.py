import openpyxl

def append_texts_to_file(xlsx_file_path, text_file_path):
    workbook = openpyxl.load_workbook(xlsx_file_path)
    sheet = workbook['Sheet1']
    with open(text_file_path, 'a', encoding='utf-8') as text_file:
        for row in sheet.iter_rows(min_row=2, values_only=True):  # Assuming 'Text' is in the first column and skipping header
            text = row[0]  # Adjust the index if 'Text' is not the first column
            if text:  # Ensure there's text to write
                text_file.write(str(text) + '\n')

if __name__ == "__main__":
    xlsx_file_path = input("Enter the path to the Excel file: ")
    text_file_path = input("Enter the path to the text file where texts will be appended: ")
    append_texts_to_file(xlsx_file_path, text_file_path)
    print("Texts have been appended to the text file.")