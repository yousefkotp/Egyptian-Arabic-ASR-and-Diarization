# ASR For Egyptian Dialect

<img src = "https://github.com/user-attachments/assets/50eddbb2-b99b-4eb9-8c15-463138ad1913" width="100%">

This repository is the submission from the Speech Squad team for the MTC-AIC 2 challenge. It contains the code for our experiments in training an Automatic Speech Recognition (ASR) model for the Egyptian dialect. We propose a novel four-stage training pipeline that enabled our model to achieve a Mean Levenshtein Distance score of `9.588644` on the test set which could be viewed as character error rate. Our model utilizes the FastConformer architecture with 32 million parameter to train and incorporates both Connectionist Temporal Classification (CTC) and Recurrent Neural Network Transducer (RNN-T). The four stages of our pipeline include pretraining on a synthetic dataset generated using GPT-4o and OpenAI's Text-to-Speech (TTS) which we publicly release, followed by training on the real dataset with CTC, further training with RNN-T, and finally fine-tuning on adaptation data. This comprehensive approach allowed us to maximize the model's performance and adaptability to the Egyptian Arabic dialect.

- **Note:** This README file is covering the work done for ASR task. For the work done in Diarization please refer to [Diarization Docs](https://github.com/AbdelrhmanElnenaey/ASR_for_egyptian_dialect/blob/main/Documentation/Diarization_README.md) README file. 

- **Note:** To run the diarization along with the transcriptions (reproduce the results) please refer to [Inference_README](https://github.com/AbdelrhmanElnenaey/ASR_for_egyptian_dialect/blob/main/inference/README.md) and [Inference](https://github.com/AbdelrhmanElnenaey/ASR_for_egyptian_dialect/tree/main/inference) section.

## Table of Contents
- [ASR For Egyptian Dialect](#asr-for-egyptian-dialect)
  * [Installation Guide](#installation-guide)
    + [Clone the repository](#clone-the-repository)
    + [Install and setup virtual environment](#install-and-setup-virtual-environment)
    + [Activate the virtual environment](#activate-the-virtual-environment)
    + [Install the required packages](#install-the-required-packages)
    + [Installing other requirements](#installing-other-requirements)
    + [Dataset Download & Setup](#dataset-download---setup)
  * [Dataset](#dataset)
    + [Real](#real)
    + [Synthetic](#synthetic)
    + [Data Filteration](#data-filteration)
  * [Tokenizer](#tokenizer)
  * [Preprocessing](#preprocessing)
  * [Data Augmentation](#data-augmentation)
    + [Spectrogram Augmentation](#spectrogram-augmentation)
    + [Dithering](#dithering)
    + [Exploration of Other Data Augmentation](#exploration-of-other-data-augmentation)
  * [Synthetic Dataset Generation](#synthetic-dataset-generation)
    + [Generate Text Corpus using OpenAI](#generate-text-corpus-using-openai)
    + [Converting Text File into CSV](#converting-text-file-into-csv)
    + [Generate Synthetic Dataset using OpenAI TTS](#generate-synthetic-dataset-using-openai-tts)
    + [Resultant Synthetic Dataset](#resultant-synthetic-dataset)
  * [Chosen Architecture](#chosen-architecture)
  * [Training](#training)
    + [1. Pretraining FastConformer-CTC on Synthetic Data](#1-pretraining-fastconformer-ctc-on-synthetic-data)
    + [2. Training FastConformer-CTC on Real Data](#2-training-fastconformer-ctc-on-real-data)
    + [3. Training FastConformer-Transducer on Real Data](#3-training-fastconformer-transducer-on-real-data)
    + [4. Fine-tuning FastConformer-Transducer on Adaptation Data](#4-fine-tuning-fastconformer-transducer-on-adaptation-data)
    + [Learning Rate Schedule](#learning-rate-schedule)
  * [Inference](#inference)
    + [Example Usage](#example-usage)
    + [Changes in Decoding Strategy](#changes-in-decoding-strategy)
  * [Insights](#insights)
  * [Example Usage for Other Functionalities](#example-usage-for-other-functionalities)
    + [Generate Speaker Embeddings](#generate-speaker-embeddings)
    + [Generate Manifest File](#generate-manifest-file)
    + [Generate Labels](#generate-labels)
  * [Contributors](#contributors)
  * [Supervisor](#supervisor)
  * [References](#references)


## Installation Guide

### Clone the repository

```bash
git clone https://github.com/AbdelrhmanElnenaey/ASR_for_egyptian_dialect
cd ASR_for_egyptian_dialect
```

### Install and setup virtual environment

```bash
pip install virtualenv
virtualenv -p python3.10 venv
```

### Activate the virtual environment

- If you are using Windows:
```bash
.\venv\Scripts\activate
```

- If you are using Linux or MacOS:
```bash
source venv/bin/activate
```
### Install the required packages

```bash
pip install -r requirements.txt
```

### Installing other requirements
```bash
pip install boto3 --upgrade
pip install text-unidecode
python -m pip install git+https://github.com/NVIDIA/NeMo.git@r2.0.0rc0#egg=nemo_toolkit[asr]
sudo apt-get install -y sox libsndfile1 ffmpeg
```

- Please note that `sudo apt-get install -y sox libsndfile1 ffmpeg` command only work for `Debian-based Linux distributions`. For windows and MacOS, you can install `sox` and `ffmpeg` using the following links:
  - [SoX](http://sox.sourceforge.net/)
  - [FFmpeg](https://ffmpeg.org/)


### Dataset Download & Setup
Before downloading the datasets, ensure you have sufficient storage space available (~25 GB) and you are connected to a stable internet connection. Datasets are large and may take time to download.

To download the datasets both real and synthetic from Google Drive, run the following commands:
```python
python data/download_datasets.py
```

After downloading the datasets, you have to build the manifest files by running the following command:
```python
python data/build_manifest.py
```

## Dataset

### Real
The competition organizers provided a relatively small dataset containing ~50,000 samples of speech-transcript paired data which will be used for training and provided another dataset for adaptation consisting of ~2200 samples. The dataset is in the form of a CSV file containing the following columns:
- `audio`: the name of the audio file
- `transcript`: the transcript of the audio file

### Synthetic

Since the dataset provided by the challenge organizers is very small, we decided to generate a synthetic dataset.

We utilized GPT-4o provided by OpenAI to generate a synthetic Egyptian text corpus using through API. These sentences generated by GPT-4o LLM later goes through OpenAI TTS model through another API to produce synthetic data that looks like the real samples. We sample the speed and speaker randomly to result in diverse dataset. The synthetic dataset contains roughly 30,000 sample of speech-transcript paired data. Since the synthetic data is not perfect, it is used in the pretraining phase to improve the model's performance before fine-tuning on the real data provided by the competition.

This data help the model capture plain Arabic phonemes in pretraining phase before finetuning it. We publicly released the synthetic dataset and can be found [Google Drive](https://drive.google.com/drive/folders/1jRb0X9_O6p6UOpIyZ2NoxF1_mjYbty4M?usp=sharing).

[Play First Audio Sample](assets/00d5ab4304518f11.wav)

[Play Second Audio Sample](assets/00b906eba080054a.wav)


### Data Filteration
To standarize the data, we normalized the transcripts. We follow the pipeline introduced by [ArTST: Arabic Text and Speech Transformer](https://arxiv.org/abs/2310.16621). All punctuation marks were removed with the exception of `@` and `%`. Additionally, all diacritics were removed, and Indo-Arabic numerals were replaced with Arabic numerals to ensure uniformity. The vocabulary is comprised of individual Arabic alphabets, numerals, and select English characters from the training dataset, in addition to some special characters like `@` and `%`. For speech data, we standardized the sampling rate to be 16 kHz across all collected datasets. An updated version of the `csv` files can be found in the `data` directory. Normalized transcripts were filtered to exclude foreign characters and only include the allowed characters in the `data/allowed_chars.txt` file.

- To normalize the transcripts, run the following command:

```bash
python utils/transcript_normalization.py --input_file=data/train.csv --output_file=data/train_artst_normalized.csv
python utils/transcript_normalization.py --input_file=data/adapt.csv --output_file=data/adapt_artst_normalized.csv
```

- To filter the normalized transcripts which execlude foreign characters or charachters not found in `data/allowed_chars.txt` file, run the following command:

```bash
python utils/execlude_foreign_characters.py --input_file=data/train_artst_normalized.csv --output_file=data/train_artst_normalized_filtered.csv --char_file=allowed_chars.txt  
python utils/execlude_foreign_characters.py --input_file=data/adapt_artst_normalized.csv --output_file=data/adapt_artst_normalized_filtered.csv --char_file=allowed_chars.txt  
```

## Tokenizer
We use a tokenizer to convert the transcripts into tokens. We use the `bpe` tokenizer obtained from `process_asr_text_tokenizer.py` script provided by NVIDIA NeMo. The tokenizer is trained on the training dataset provided by the competition and is saved in the `tokenizer` folder with a vocab size of `256`. 

We explored different tokenizer obtained through `SentencePiece` with a vocab size of 32,000 where we gathered Egyptian text corpus on the internet (Reddit, Twitter, EL-youm Elsabe', and Wikipedia) and trained the tokenizer on 63 million egyptian word (provided on demand due to large size). However, the tokenizer did not perform well on the dataset provided by the competition.

We believe that there is room for improvement in the tokenizer where we plan to explore different tokenizers and vocab sizes in the future.

## Preprocessing
To extract features from the raw audio data which is used later by FastConformer model, we use the `MelSpectrogram` feature extractor provided by NVIDIA NeMo. The feature extractor is used to convert the raw audio data into Mel spectrograms. The Mel spectrograms are then normalized and augmented using various techniques to improve the model's performance. We mainly configure the feature extractor with the following parameters:

- `sample rate`: The sample rate of the audio data. We set this parameter to `16000` Hz as in the training data provided by the competition.
- `normalize`: Whether to normalize the Mel spectrograms. We set this parameter to `per_feature` to normalize each feature independently.
- `window size`: The size of the window used to compute the Mel spectrograms. We set this parameter to `0.025` seconds.
- `window stride`: The stride (in seconds) between successive windows during STFT (Short Time Fourier Transform). We used a stride of `0.01` seconds.
- `window`: The type of window function applied to each audio frame before computing the Fourier transform. "hann" specifies the Hann window, which helps minimize the spectral leakage.
- `features`: The number of Mel frequency bands (or features) to generate. We used 80 features, which is typical for ASR tasks.
- `n_fft`: The number of points used in the FFT (Fast Fourier Transform) to calculate the spectrogram. A value of 512 is used throughout the whole experiments.

## Data Augmentation
To mitigate the effects of overfitting and improve the model's robustness given limited training data, we employed data augmentation techniques. More specifically, we use spectogram augmentation and dithering.

### Spectrogram Augmentation
We used Spectrogram augmentation which is a technique used to make the model more robust by adding variability to the training data. This method modifies the spectrograms of the audio inputs to simulate variations that could occur in real-world data. We mainly tune four hyperparameters:

- `Frequency Masks`: The number of frequency masks to apply. This parameter controls the number of frequency channels to mask.
- `Time Masks`: The number of time masks to apply. This parameter controls the number of time steps to mask.
- `Freq Width`: The width of the frequency mask, which is 27. This defines the number of frequency channels to mask.
- `Time Width`: The width of the time mask, which is 0.05. This defines the proportion of the time axis to mask.

Those parameters are changed depending on each phase of training phases as explained in [Training](#training) section.

### Dithering
Dithering is a technique used in digital signal processing to add a low level of noise to an audio signal. This noise can help mask quantization errors and make the audio signal more robust. dithering helped us in improving the generalization of the model. We mainly set dithering = 0.00001 in all of our experiments.

### Exploration of Other Data Augmentation
We explored other data augmentation techniques as reverberation which simulates the effects of audio reverberating in various environments. Moreover, we explored noise perturbation which refers to the addition of synthetic noise to an audio signal, in our case, we used additive white noise. However, we found that these techniques prevented the model from learning the training data effectively and did not improve the model's performance. We believe that excess data augmentation did not help the model in learning the training data effectively.

## Synthetic Dataset Generation

### Generate Text Corpus using OpenAI

The text corpus was generated using a custom Python script that leverages the OpenAI API, specifically utilizing the GPT-4o (omni) model. This approach allowed us to create a diverse and rich set of sentences in Egyptian Arabic, reflecting various aspects of local culture, everyday life, and a wide range of topics. The script ensures the uniqueness of each sentence, focusing on clarity, naturalness, and the inclusion of colloquial expressions to add authenticity.


The script `utils/generate_text_corpus.py` performs the following steps to generate the text corpus:

1. **Initialization**: It sets up the OpenAI client using an API key stored in the environment variables. Please make sure to set `OPENAI_API_KEY` environment variable to your OpenAI API key.
2. **Dynamic Prompting**: For each request, it sends a dynamic prompt to the GPT-4o model. The prompt includes instructions to generate sentences in Egyptian Arabic, covering diverse topics and ensuring each sentence is unique and clear.
3. **Sentence Generation**: The model generates responses based on the prompt. Each response is then split into individual sentences.
4. **Uniqueness and Quality Checks**: The script filters out any sentence that is either too short (less than two words) or already present in the set of unique sentences.
5. **Output**: Unique sentences are written to an output file, `output_sentences.txt`, with each sentence on a new line.

This process repeats until the script reaches the target number of unique sentences, which is set to 30,000 in this case.


To generate the text corpus, **ensure you have set the `OPENAI_API_KEY` environment variable to your OpenAI API key.** Then, run the script from your terminal:

```bash
python utils/generate_text_corpus.py
```

### Converting Text File into CSV
The output text file is then converted into a CSV file by adding `transcript` column which is added to the beginning of file. The `.txt` extension is replaced with `.csv` extension too.

### Generate Synthetic Dataset using OpenAI TTS

To generate audio from text transcripts, we developed a Python script named `utils/TTS_OpenAI.py`. This script utilizes the OpenAI API to convert text transcripts into speech, simulating various voices and adjusting the speech speed for a more natural and diverse audio output. The generated audio files are saved in a specified output directory, and a CSV file is updated incrementally with mappings between the audio files and their corresponding transcripts.

The script performs the following steps to generate audio from text:

1. **Reading Transcripts**: It reads transcripts from a CSV file specified by the user. The CSV file must contain a column named 'transcript' with the text intended for speech conversion.
2. **Voice Selection**: For each transcript, the script randomly selects a voice from a predefined list of voices provided by the OpenAI API. This adds variety to the audio output. The available voices are `alloy`, `echo`, `fable`, `onyx`, `nova`, and `shimmer`.
3. **Speed Variation**: The speech speed is uniformly sampled between 0.8 and 1.2 times the normal speed, introducing natural variation in the speech tempo.
4. **Audio Generation**: The OpenAI API's text-to-speech model is invoked with the selected voice, transcript, and speed to generate the speech audio in WAV format.
5. **Sample Rate Adjustment**: If the desired sample rate differs from the default (24,000 Hz), the script uses `pydub` to adjust the sample rate of the generated audio file to be 16,000 Hz, ensuring compatibility with the ASR model's requirements.
6. **Output Mapping**: A mapping of audio file names to their corresponding transcripts is appended to a CSV file (`synthetic.csv`) in the output directory.

To use the script, you need to provide the path to the CSV file containing the transcripts, the output directory for the audio files, and optionally, the desired sample rate for the audio files. **Ensure you have set the `OPENAI_API_KEY` environment variable to your OpenAI API key.** Then, run the script from your terminal:

```bash
python TTS_OpenAI.py --csv_file output_sentences.csv --output_dir data/synthetic --sample_rate 16000
```

where:
- `--csv_file`: The path to the CSV file containing the transcripts (generated text corpus after adding `transcript` column).
- `--output_dir`: The path to the output directory where the audio files will be saved.
- `--sample_rate`: The desired sample rate for the audio files.

### Resultant Synthetic Dataset

You can find the [synthetic.csv](data/synthetic.csv) file containing the generated transcripts and their corresponding audio files in the `data` directory. Also don't forget to download the audio files from [Google Drive](https://drive.google.com/drive/folders/1jRb0X9_O6p6UOpIyZ2NoxF1_mjYbty4M?usp=sharing).

## Chosen Architecture
Conformer-based models have shown great performance in end-to-end automatic speech recognition task. Due to their encoder architecture that integrates depth-wise convolutional layers for local features and self-attention layers for global context, conformers have gained widespread adoption in industry, particularly for real-time streaming ASR applications both on-device and in cloud environments.

To boost the Conformer model's efficiency, several key changes were made to obtain FastConformer: an 8x downsampling at the encoder's start reduced subsequent attention layers' computational load by 4x. Convolutional sub-sampling layers were replaced with depthwise separable convolutions, downsampling filters were cut to 256, and kernel size was reduced to 9. These adjustments aimed to enhance efficiency while maintaining/improving model performance.

We decided to use FastConformer for its fast (near real-time) inference speed without incurring any compromises in its performance. The inference speed criteria have allowed us to perform fast-paced research iterations.

## Training
The intuition behind using a four-stage pipeline for training the FastConformer model on Egyptian Arabic ASR stems from the idea to gradually and effectively adapt the model to the complexities of the Egyptian language **given limited data**. The four stages which are: pretraining on synthetic data, training on real data with CTC, training on real data with RNN-T, and fine-tuning on adaptation data. Each serve a distinct purpose in refining the model.

### 1. Pretraining FastConformer-CTC on Synthetic Data
Starting with pretraining on synthetic data, we aim to provide the model with a broad and diverse exposure to the phonetic patterns and acoustic variations in Egyptian Arabic. Synthetic data, generated using OpenAI's GPT-4o and TTS help the model learn fundamental phonetic structures. This stage helps initialize the model's parameters in a meaningful way, establishing a robust foundation that aids in better generalization during subsequent stages.

In this stage, we maximized data augmentation because the data generated were easy to learn by Fast Conformer model. The exact training cofinguration can be found in [pretrain-ctc.yaml](configs/pretrain-ctc.yaml) file.

To train the model on this stage, run the following command:

```bash
python train/fast_conformer_ctc_pretrain.py 
```

### 2. Training FastConformer-CTC on Real Data
The second stage, training on real data with Connectionist Temporal Classification (CTC), is crucial for further refining the model's understanding of natural speech. The CTC loss function is particularly effective for sequence-to-sequence tasks where the alignment between input (audio) and output (transcription) is not known. By focusing initially on CTC, we allow the model to learn a reliable alignment and decoding process, improving its capability to handle varying lengths of input sequences and their corresponding transcriptions. This stage builds on the model's pretraining on synthetic data, enhancing its ability to recognize and transcribe Egyptian Arabic speech patterns specially for the alignment between the audio and the transcript which is mainly not captured in Transducer.

The exact training cofinguration can be found in [train-ctc.yaml](configs/train-ctc.yaml) file.

To train the model on this stage, run the following command:

```bash
python train/fast_conformer_ctc_train.py --checkpoint_path "/path/to/your/checkpoint.ckpt"
```

Where:
- `--checkpoint_path`: The path to the checkpoint file from the previous stage.

### 3. Training FastConformer-Transducer on Real Data
Transitioning to the third stage, we transfer the encoder learnt from previous two pipelines and uses new decoder and then we train the while model using the Recurrent Neural Network Transducer (RNN-T) loss on real data. The RNN-T loss function is designed to better handle the temporal dependencies inherent in speech data which is not captured in CTC. This stage builds on the model's initial alignment learned during the CTC phase, enhancing its ability to accurately predict sequences and further refining its performance by leveraging the temporal structure of the transcript.

The exact training cofinguration can be found in [train-transducer.yaml](configs/train-transducer.yaml) file.

To train the model on this stage, run the following command:

```bash
python train/fast_conformer_transducer_train.py --checkpoint_path "/path/to/your/checkpoint.ckpt"
```

Where: 
- `--checkpoint_path`: The path to the checkpoint file from the previous stage whose encoder will be transferred to the new model.

### 4. Fine-tuning FastConformer-Transducer on Adaptation Data
Finally, the fine-tuning stage on adaptation data ensures that the model can adapt to specific characteristics or distributions that may be unique to the test set. It is worth noting that the model is fine-tuned on both train and adapt dataset due to limited number of samples in the adapt dataset alone which would lead to overfitting. This stage allows the model to refine its predictions and improve its performance on the test set by learning from the adaptation data.

The exact training cofinguration can be found in [adapt-transducer](configs/adapt-transducer.yaml) file.

To train the model on this stage, run the following command:

```bash
python train/fast_conformer_transducer_finetune.py --checkpoint_path "/path/to/your/checkpoint.ckpt"
```

Where:
- `--checkpoint_path`: The path to the checkpoint file from the previous stage.

This phased approach, from synthetic data pretraining to targeted fine-tuning, ensures that the model is well-prepared to handle the complexities of Egyptian Arabic ASR with high accuracy given limited training data.

- **Note:** Every checkpoint for each one of the stages can be found in this [Google Drive link](https://drive.google.com/drive/folders/1bQ-k6o9B7qlvNO6vujZGnpf2XwI9V8zB?usp=sharing). It is highly advised to proceed with only the last checkpoint of the last stage (`asr_model.ckpt`) if you want to further fine-tune it infere with it which could be found [here](https://drive.google.com/file/d/1faLSvzXVcZd_lvBXxxdWYyBGyGnC2ijL/view?usp=sharing).

### Learning Rate Schedule
We use Cosine Annealing learning rate schedule for all training phases. This schedule is effective in preventing the model from overshooting while converging fast. The learning rate is gradually decreased over the course of training, allowing the model to explore a wider range of solutions and converge to a better optimum. This was extremly helpful during the last stage of training where the model was able to learn more from the adaptation data.

## Inference
To replicate our inference results (ASR only), `inference.py` is provided.

The script downlads the checkpoints from google drive if it is not downloaded, transcribes audio files found in `data_dir` using specified `asr` model and outputs the results in `csv format`.

The checkpoint can be found [here](https://drive.google.com/file/d/1faLSvzXVcZd_lvBXxxdWYyBGyGnC2ijL/view?usp=sharing).
### Example Usage
```bash
python inference.py --asr_model asr_model.ckpt \
                    --data_dir data/test \
                    --output results.csv
```
For more information, use `inference.py -h`. Feel free to write name of a checkpoint that doesn't exist yet, the script will download it for you.

### Changes in Decoding Strategy
During inference, we change the decoding strategy from `greedy` to `beam` with `beam_size=5` to improve the model's performance. This further improves results by considering multiple hypotheses during decoding not only the most probable one. `greedy` decoding is only used during training for all phases to speed up the training process and reduce the computational cost.

## Insights
- BPE Tokenizer vs. Unigram Tokenizer
  - The BPE (Byte Pair Encoding) tokenizer outperformed the unigram tokenizer in our experiments.
  - Our interpretation BPE tokenizer's ability to handle subword units more effectively might have contributed to better performance, especially given the diverse phonetic structure of the Egyptian Arabic dialect.


- Performance Variability with Tokenizer Choice and Vocabulary Size
  - The performance of our model varied significantly based on the choice of tokenizer and the size of the vocabulary.
  - This suggests that the right combination of tokenizer and vocabulary size is crucial for optimizing ASR performance, indicating a need for careful experimentation and tuning in these areas.

- Effectiveness of Synthetic Data for Pretraining
  - Pretraining on synthetic data improved the model's performance, even though the data was not the most realistic.
  - Our interpretation suggests that synthetic data helped the model to capture basic Arabic phonemes and provided a good initial learning phase, which was crucial given the small size of the real dataset.

- Training Challenges with Fast Conformer
  - The Fast Conformer model required a large number of epochs to start converging.
  - This could be attributed to the complexity and size of the dataset. The small and challenging nature of the data might have made it difficult for the model to learn patterns quickly.

- Faster Convergence on Synthetic Data
  - The model converged much faster on the synthetic data compared to the real dataset.
  - Our interpretation is that the synthetic data, being more consistent and possibly less noisy, allowed the model to learn more efficiently in the initial phases of training.

- Too much augmentation is terrible
  - We found that excessive data augmentation, such as reverberation and noise perturbation, did not improve the model's performance. Moreover, the more we incread spectrogram augmentation, the more the model is not able to learn anything at all where WER will be very high and not able to decrease.
  - This suggests that a balance in data augmentation is crucial to prevent overfitting and ensure effective learning especially when the data is limited.

- Importance of Fine-tuning on Adaptation Data
  - Fine-tuning on adaptation data was crucial for improving the model's performance on the test set.
  - Our interpretation is that the adaptation data comes from the same distribution as the test set, allowing the model to learn specific characteristics that are essential for accurate predictions which enabled such high accuracy on the test set.

- CTC vs. RNN-T Loss Functions
  - The CTC loss function was effective in the initial stages of training, allowing the model to learn alignment. However, it reached a plateau in performance where the model was not able to learn more from the data. Moreover, the model was mostly outputing jebberish words due to not taking into account the temporal dependencies in the data (context).
  - The RNN-T loss function, on the other hand, was more effective in capturing the temporal dependencies in the data, leading to better performance in the later stages of training.

- Importance of Decoding Strategy
  - `Beam` strategy with `beam_size=5` boosted the performance much more than the `greedy` strategy. However, it is not computationally feasible to train the model with `beam` strategy due to the high computational cost.
  - This suggests that considering multiple hypotheses during decoding is crucial for improving the model's performance but should not be used during training due to the high computational cost.

- A new state-of-the-art for Arabic speech enhancement is needed
  - The current state-of-the-art models for Arabic speech enhancement are not well-suited for the Egyptian dialect, which has unique phonetic and linguistic characteristics.
  - We explored different noise removal models, the best one was NVIDIA'S [CleanUNet](https://github.com/NVIDIA/CleanUNet) which was meant to remove the noise and enhance the speech. However, the model did not perform well on the Egyptian dialect test dataset and the ASR performed worse when this speech enhancer was used.
  - We believe that developing a new state-of-the-art model for speech enhancement **specifically for the Egyptian dialect** is essential to improve ASR performance and accuracy.

## Example Usage for Other Functionalities

### Generate Speaker Embeddings
```bash
python utils/generate_speaker_embedding.py --input_dir=data/train --output_dir=data/speaker_embedding/train
python utils/generate_speaker_embedding.py --input_dir=data/adapt --output_dir=data/speaker_embedding/adapt
```

### Generate Manifest File
```bash
python utils/generate_manifest_file.py --audio_dir=data/train --embedding_dir=data/speaker_embedding/train --output_file=data/train.tsv --audio_csv=data/train_artst_normalized_filtered.csv
python utils/generate_manifest_file.py --audio_dir=data/adapt --embedding_dir=data/speaker_embedding/adapt --output_file=data/adapt.tsv --audio_csv=data/adapt_artst_normalized_filtered.csv
```

### Generate Labels
```bash
python utils/generate_label_file.py --tsv_file=data/train.tsv --csv_file=data/train_artst_normalized_filtered.csv --output_file=data/train_labels.txt
python utils/generate_label_file.py --tsv_file=data/adapt.tsv --csv_file=data/adapt_artst_normalized_filtered.csv --output_file=data/labels/adapt.txt
```

## Contributors
- [Yousef Kotp](https://github.com/yousefkotp)
- [Karim Alaa](https://github.com/Karim19Alaa)
- [Abdelrahman Elnenaey](https://github.com/AbdelrhmanElnenaey)
- [Rana Barakat](https://github.com/ranabarakat)
- [Louai Zahran](https://github.com/LouaiZahran)

## Supervisor
- [Ismail El-Yamany](https://github.com/IsmailElYamany)

## References
- [FastConformer: Efficient and Accurate Conformer ASR with Low Latency](https://arxiv.org/abs/2305.05084)
- [Conformer: Convolution-augmented Transformer for Speech Recognition](https://arxiv.org/abs/2005.08100)
- [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477)
- [Arabic - Egyptian comparable Wikipedia corpus](https://www.kaggle.com/datasets/mksaad/arb-egy-cmp-corpus)
- [ElevenLabs: Text to Speech & AI Voice Generator](https://elevenlabs.io/)
- [GPT-4o](https://openai.com/index/hello-gpt-4o/)
- [OpenAI TTS](https://platform.openai.com/docs/guides/text-to-speech)
