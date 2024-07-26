import argparse
import torch
import soundfile as sf
import os
import gdown
import nemo.collections.asr as nemo_asr
from ruamel.yaml import YAML
from omegaconf import OmegaConf
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_asr_model(ckpt_path):
    if(not os.path.exists(ckpt_path)):
        url = "https://drive.google.com/file/d/1faLSvzXVcZd_lvBXxxdWYyBGyGnC2ijL/view?usp=sharing"
        gdown.download(url, ckpt_path, quiet=False, fuzzy=True)

    config_path = 'configs/FC-transducer-inference.yaml'
    yaml = YAML(typ='safe')
    with open(config_path) as f:
        params = yaml.load(f)
    params['model'].pop('test_ds')
    conf = OmegaConf.create(params)

    model = nemo_asr.models.EncDecRNNTBPEModel(cfg=conf['model']).to(device)
    model.load_state_dict(torch.load(ckpt_path)['state_dict'])
    model.eval()
    return model

def create_parser():
    parser = argparse.ArgumentParser(description="ASR Inference")
    parser.add_argument("--asr_model", type=str, help="Path to the ASR model checkpoint", default="asr_model.ckpt")
    parser.add_argument("--data_dir", type=str, help="Path to the directory containing test data", default="data/adapt")
    parser.add_argument("--output", type=str, help="Path to the output file", default="results.csv")
    return parser

def infere(model, audio):
    return model.transcribe([audio])

if __name__ == "__main__":
    args = create_parser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = args.data_dir
    asr_model = load_asr_model(args.asr_model)

    with open(args.output, "w+", encoding='utf-8') as fp:
        fp.write("audio,transcript,audio_duration,transcription_time,rtf\n")

    total_audio_duration = 0
    total_transcription_time = 0
    file_count = 0

    for filename in os.listdir(data_dir):
        audio, sr = sf.read(os.path.join(data_dir, filename), dtype='float32')
        audio_duration = len(audio) / sr 

        start_time = time.time()
        with torch.no_grad():
            rv = infere(model=asr_model, audio=audio)
        end_time = time.time()

        transcription_time = end_time - start_time
        rtf = transcription_time / audio_duration

        total_audio_duration += audio_duration
        total_transcription_time += transcription_time
        file_count += 1

        with open(args.output, "a+") as fp:
            clean_output = rv[0][0]
            fp.write(f"{os.path.splitext(os.path.basename(filename))[0]},{clean_output},{audio_duration:.2f},{transcription_time:.2f},{rtf:.2f}\n")

    average_rtf = total_transcription_time / total_audio_duration

    print(f"\nProcessed {file_count} files")
    print(f"Total audio duration: {total_audio_duration:.2f} seconds")
    print(f"Total transcription time: {total_transcription_time:.2f} seconds")
    print(f"Average Real-Time Factor: {average_rtf:.2f}")
