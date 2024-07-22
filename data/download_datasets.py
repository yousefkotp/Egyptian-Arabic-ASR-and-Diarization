import os
import gdown
import zipfile

def download_and_unzip(gdrive_url, download_path, extract_to):
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    if not os.path.exists(extract_to):
        os.makedirs(extract_to)

    temp_file_path = os.path.join(download_path, 'temp_downloaded_file')
    zip_file_path = os.path.join(download_path, 'downloaded_file.zip')
    gdown.download(gdrive_url, temp_file_path, quiet=False, fuzzy=True)

    os.rename(temp_file_path, zip_file_path)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_file_path)

    print(f"Downloaded and extracted to {extract_to}")

print("Downloading Synthetic Dataset...")
gdrive_url = "https://drive.google.com/file/d/1-NH0n9jXRdXIa8sSA5Iyk1sa7tGLqFeh/view?usp=sharing"
download_path = 'data'
extract_to = 'data'
download_and_unzip(gdrive_url, download_path, extract_to)
print("-" * 50)

print("Downloading Real Dataset...")
gdrive_url = "https://drive.google.com/file/d/154GwOvpgk-MPyQVLBYRGvg85FetxYBJo/view?usp=sharing"
download_path = 'data'
extract_to = 'data'
download_and_unzip(gdrive_url, download_path, extract_to)
print("-" * 50)


print("Downloading test dataset...")
gdrive_url = "https://drive.google.com/file/d/1hSQriqeMZVOXptzr623qY9-nRB2TZmhR/view?usp=sharing"
download_path = 'data'
extract_to = 'data'
download_and_unzip(gdrive_url, download_path, extract_to)
print("-" * 50)
print("Finished downloading all required datasets.")
