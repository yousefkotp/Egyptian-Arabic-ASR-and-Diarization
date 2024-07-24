import argparse
import os
import json
import torch
import soundfile as sf
import nemo.collections.asr as nemo_asr
import gdown
import subprocess
import logging
from omegaconf import OmegaConf, open_dict
from ruamel.yaml import YAML
from pydub import AudioSegment
from nemo.collections.asr.parts.utils.diarization_utils import OfflineDiarWithASR
import wget


def load_asr_model(ckpt_path,config_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if not os.path.exists(ckpt_path):
        url = "https://drive.google.com/file/d/1faLSvzXVcZd_lvBXxxdWYyBGyGnC2ijL/view?usp=drive_link"
        gdown.download(url, ckpt_path, quiet=False, fuzzy=True)

    # config_path = 'configs/FC-transducer-inference.yaml'
    yaml = YAML(typ='safe')
    with open(config_path) as f:
        params = yaml.load(f)
    params['model'].pop('test_ds')
    conf = OmegaConf.create(params)

    model = nemo_asr.models.EncDecRNNTBPEModel(cfg=conf['model']).to(device)
    decoding_cfg = model.cfg.decoding
    with open_dict(decoding_cfg):
        decoding_cfg.preserve_alignments = True
        decoding_cfg.compute_timestamps = True
        decoding_cfg.strategy = 'greedy'
        model.change_decoding_strategy(decoding_cfg)
    model.load_state_dict(torch.load(ckpt_path)['state_dict'])
    model.eval()
    return model

def create_parser():
    parser = argparse.ArgumentParser(description="ASR Inference")
    parser.add_argument("--asr_model", type=str, help="Path to the ASR model checkpoint", default="asr_model.ckpt")
    parser.add_argument("--data_dir", type=str, help="Path to the directory containing test data", default="data/adapt")
    parser.add_argument("--asr_output", type=str, help="Path to the asr output file", default="results.csv")
    parser.add_argument("--input_manifest_path", type=str, help="Path to the manifest file before preprocessing", default="test_manifest.json")
    parser.add_argument("--output_manifest_path", type=str, help="Path to the output manifest after preprocessing", default="test_manifest_vocals.json")
    parser.add_argument("--temp_output_dir", type=str, help="Path to a temp output dir (needed for preprocessing)", default="temp_outputs")
    parser.add_argument("--mono_output_dir", type=str, help="Path to the output dir after preprocessing and converting to mono", default="temp_outputs_mono")
    parser.add_argument("--config_path", type=str, help="Path to ASR config file", default="configs/FC-transducer-inference.yaml")
    
    return parser

def infere(model, audio):
    hypotheses = model.transcribe([audio], return_hypotheses=True)
    if type(hypotheses) == tuple and len(hypotheses) == 2:
        hypotheses = hypotheses[0]
    return hypotheses

def preprocess_manifest(input_manifest_path, output_manifest_path, temp_output_dir, mono_output_dir):
    os.makedirs(mono_output_dir, exist_ok=True)
    with open(input_manifest_path, 'r') as f:
        manifest = [json.loads(line) for line in f]

    new_manifest = []
    for entry in manifest:
        audio_filepath = entry['audio_filepath']
        base_name = os.path.splitext(os.path.basename(audio_filepath))[0]

        try:
            result = subprocess.run(
                [
                    "python3",
                    "-m",
                    "demucs.separate",
                    "-n",
                    "htdemucs",
                    "--two-stems=vocals",
                    audio_filepath,
                    "-o",
                    temp_output_dir
                ],
                check=True,
                capture_output=True,
                text=True
            )
            original_vocal_filepath = os.path.join(temp_output_dir, "htdemucs", base_name, "vocals.wav")
            vocal_filepath = os.path.join(temp_output_dir, "htdemucs", f"{base_name}.wav")
            if os.path.exists(original_vocal_filepath):
                os.rename(original_vocal_filepath, vocal_filepath)
            else:
                raise FileNotFoundError(f"Expected output file not found: {original_vocal_filepath}")
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logging.warning(f"Source splitting failed for {audio_filepath}, using original audio file. Error: {str(e)}")
            vocal_filepath = audio_filepath

        sound = AudioSegment.from_file(vocal_filepath).set_channels(1)
        mono_vocal_filepath = os.path.join(mono_output_dir, f"{base_name}.wav")
        sound.export(mono_vocal_filepath, format="wav")

        new_entry = entry.copy()
        new_entry['audio_filepath'] = mono_vocal_filepath
        new_manifest.append(new_entry)

    with open(output_manifest_path, 'w') as f:
        for entry in new_manifest:
            json.dump(entry, f)
            f.write('\n')

def run_asr_inference(data_dir, asr_model, output_file):
    word_hyp = {}
    word_ts_hyp = {}

    with open(output_file, "w+", encoding='utf-8') as fp:
        fp.write("audio,transcript,start,end\n")

    for filename in os.listdir(data_dir):
        if filename.endswith('.wav'):
            audio, sr = sf.read(os.path.join(data_dir, filename), dtype='float32')
            audio_path = os.path.join(data_dir, filename)
            with torch.no_grad():
                rv = infere(model=asr_model, audio=audio_path)

            timestamp_dict = rv[0].timestep
            time_stride = 8 * asr_model.cfg.preprocessor.window_stride
            word_timestamps = timestamp_dict['word']

            words = []
            word_timestamps_list = []

            with open(output_file, "a+") as fp:
                for i, stamp in enumerate(word_timestamps):
                    start = stamp['start_offset'] * time_stride
                    end = stamp['end_offset'] * time_stride
                    transcription = stamp['word']

                    words.append(transcription)
                    word_timestamps_list.append([start, end])

                    fp.write(f"{filename},{transcription},{start},{end}\n")
                unique_id = os.path.splitext(filename)[0]
                word_hyp[unique_id] = words
                word_ts_hyp[unique_id] = word_timestamps_list

    return word_hyp, word_ts_hyp

def run_diarization(manifest_file, word_ts_hyp, word_hyp):
    
    
    data_dir = 'diarization_output'
    os.makedirs('diarization_output',exist_ok=True)
    
    DOMAIN_TYPE = "general" 
    CONFIG_FILE_NAME = f"diar_infer_{DOMAIN_TYPE}.yaml"

    CONFIG_URL = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{CONFIG_FILE_NAME}"

    if not os.path.exists(os.path.join(data_dir,CONFIG_FILE_NAME)):
        CONFIG = wget.download(CONFIG_URL, data_dir)
    else:
        CONFIG = os.path.join(data_dir,CONFIG_FILE_NAME)
        
    cfg = OmegaConf.load(CONFIG)
    cfg.diarizer.collar = 0
    cfg.diarizer.vad.parameters.window_length_in_sec = 0.63
    cfg.diarizer.vad.parameters.shift_length_in_sec = 0.01
    cfg.diarizer.manifest_filepath = manifest_file
    cfg.diarizer.out_dir = 'diarization_output'
    cfg.diarizer.speaker_embeddings.model_path = 'titanet_large'
    cfg.diarizer.clustering.parameters.oracle_num_speakers = False
    cfg.diarizer.vad.model_path = 'vad_multilingual_marblenet'
    cfg.diarizer.oracle_vad = False
    cfg.diarizer.asr.parameters.asr_based_vad = False
    
    
    

    asr_diar_offline = OfflineDiarWithASR(cfg.diarizer)
    diar_hyp, _ = asr_diar_offline.run_diarization(cfg, word_ts_hyp)
    trans_info_dict = asr_diar_offline.get_transcript_with_speaker_labels(diar_hyp, word_hyp, word_ts_hyp)

    return trans_info_dict

def process_json_files(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(".json") and "gecko" not in filename:
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            with open(input_path, 'r', encoding='utf-8') as file:
                data = json.load(file)

            with open(output_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=4)

def transform_json(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        with open(input_path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)

        transformed_data = []

        for sentence in data.get('sentences', []):
            transformed_data.append({
                "start": float(sentence['start_time']),
                "end": float(sentence['end_time']),
                "speaker": int(sentence['speaker'].replace("speaker_", "")),
                "text": sentence['text']
            })
        output_path = os.path.join(output_dir, filename)
        with open(output_path, 'w', encoding='utf-8') as outfile:
            json.dump(transformed_data, outfile, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()

    asr_model = load_asr_model(args.asr_model,args.config_path)
    word_hyp, word_ts_hyp = run_asr_inference(args.data_dir, asr_model, args.asr_output)
    
    preprocess_manifest(args.input_manifest_path, args.output_manifest_path, args.temp_output_dir, args.mono_output_dir)

    trans_info_dict = run_diarization(args.output_manifest_path, word_ts_hyp, word_hyp)

    output_dir = 'diarization_VAD_ASR_results'
    os.makedirs(output_dir, exist_ok=True)
    process_json_files('diarization_output/pred_rttms', output_dir)

    final_output_dir = 'diarization_VAD_ASR_results_final'
    os.makedirs(final_output_dir, exist_ok=True)
    transform_json(output_dir,final_output_dir)
    print(f'Final outputs with desired format are located in {final_output_dir}')
