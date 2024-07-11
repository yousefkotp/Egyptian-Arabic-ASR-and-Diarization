import argparse

from noise_removal.CleanUNet import CleanUNet
import torch
import soundfile as sf
import os

import gdown

import nemo.collections.asr as nemo_asr

try:
    from ruamel.yaml import YAML
except ModuleNotFoundError:
    from ruamel_yaml import YAML


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
def load_speech_enhancer(ckpt_path):
    if(not os.path.exists(ckpt_path)):
        url = "https://drive.google.com/file/d/1GfolCiR80WqM-wkjp0j518-Y1bV371wu/view?usp=drive_link"
        gdown.download(url, ckpt_path, quiet=False, fuzzy=True)
    
    enhancement_ckpt = torch.load(ckpt_path, map_location=device)

    enhancement_params = {
        "channels_input": 1,
        "channels_output": 1,
        "channels_H": 64,
        "max_H": 768,
        "encoder_n_layers": 8,
        "kernel_size": 4,
        "stride": 2,
        "tsfm_n_layers": 5,
        "tsfm_n_head": 8,
        "tsfm_d_model": 512,
        "tsfm_d_inner": 2048
    }
    enhancement_model = CleanUNet(**enhancement_params).to(device)
    enhancement_model.load_state_dict(enhancement_ckpt['model_state_dict'])
    enhancement_model.eval()

    return enhancement_model


def load_asr_model(ckpt_path):
    if(not os.path.exists(ckpt_path)):
        url = "https://drive.google.com/file/d/1dI86BSU6sqIP8PzkHrhcc8F57DQJvG1_/view?usp=drive_link"
        gdown.download(url, ckpt_path, quiet=False, fuzzy=True)

    model = model = nemo_asr.models.EncDecCTCModelBPE.load_from_checkpoint(ckpt_path).to(device)
    model.eval()
    return model
    

def create_parser():
    parser = argparse.ArgumentParser(description="ASR Inference")
    parser.add_argument("--asr_model", type=str, help="Path to the ASR model checkpoint", default="asr_model.ckpt")
    parser.add_argument("--enhancement_model", type=str, help="Path to the speech enhancement model chekpoint", default="cleanunet.pt")
    parser.add_argument("--data_dir", type=str, help="Path to the directory containing test data", default="data/adapt")
    parser.add_argument("--output", type=str, help="Path to the output file", default="results.csv")
    return parser

def infere_enhanced(model, audio, enahncer=None):

    clean = enahncer(torch.Tensor(audio).unsqueeze(0).unsqueeze(0).to(device)).squeeze()

    return model.transcribe([clean])


def infere(model, audio, enahncer=None):    
    return  model.transcribe([audio])

if __name__ == "__main__":

    args = create_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inference_method = infere
    enhancement_model = None
    
    if args.enhancement_model is not None:
        enhancement_model = load_speech_enhancer(args.enhancement_model)
        inference_method = infere_enhanced




    data_dir = args.data_dir

    asr_model = load_asr_model(args.asr_model)

    with open(args.output, "w+", encoding='utf-8') as fp:
        fp.write("audio,transcript\n")



    for filename in os.listdir(data_dir):
        audio, sr = sf.read(os.path.join(data_dir, filename), dtype='float32')
        with torch.no_grad():
            rv = inference_method(model=asr_model, audio=audio, enahncer=enhancement_model)
        with open(args.output, "a+") as fp:
            fp.write(f"{os.path.splitext(os.path.basename(filename))[0]},{rv[0]}\n")



