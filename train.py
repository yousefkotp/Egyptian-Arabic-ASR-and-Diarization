import os
import json
from tqdm import tqdm
import sys
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import logging
import torch
import nemo
import nemo.collections.asr as nemo_asr
import argparse
from pydub import AudioSegment
import librosa

def build_manifest(data_path, output_path, split, take=-1):
    with open(output_path, "w+") as fout:
        with open(f"{data_path}/{split}.csv", "r") as fp:
            header = True
            for line in tqdm(fp):
                if header:
                    header = False
                    continue

                line = line.strip()
                data = line.split(",")
                sample_path = f"{data_path}/{data[0]}.wav"
                sample = {
                    "audio_filepath": sample_path,
                    "duration": librosa.get_duration(filename=sample_path),
                    "text": data[1]
                }
                json.dump(sample, fout, ensure_ascii=False)
                fout.write("\n")
                if take > 0:
                    take -= 1
                if take == 0:
                    break

def main(train_data_path, adapt_data_path):
    build_manifest(train_data_path, "train_manifest.json", "train")
    build_manifest(adapt_data_path, "adapt_manifest.json", "adapt")

    config_path = 'configs/fast-conformer_ctc_bpe.yaml'

    try:
        from ruamel.yaml import YAML
    except ModuleNotFoundError:
        from ruamel_yaml import YAML

    yaml = YAML(typ='safe')
    with open(config_path) as f:
        params = yaml.load(f)

    train_manifest = "train_manifest.json"
    adapt_manifest = "adapt_manifest.json"

    params['model']['train_ds']['manifest_filepath'] = train_manifest
    params['model']['train_ds']['batch_size'] = 16
    params['model']['train_ds']['num_workers'] = 4

    params['model']['validation_ds']['manifest_filepath'] = adapt_manifest
    params['model']['validation_ds']['batch_size'] = 1
    params['model']['validation_ds']['num_workers'] = 4

    params['model']['tokenizer']['dir'] = "tokenizer"
    params['model']['optim']['sched']['warmup_steps'] = 0

    params['model']['optim']['lr'] = 5e-5
    params['model']['optim']['weight_decay'] = 0
    params['model']['optim']['sched']['min_lr'] = 1e-7

    params['model'].pop('test_ds')

    logging.getLogger('nemo_logger').setLevel(logging.ERROR)

    trainer = pl.Trainer(max_epochs=400, accumulate_grad_batches=16, check_val_every_n_epoch=2)

    conf = OmegaConf.create(params)

    model = nemo_asr.models.EncDecCTCModelBPE(cfg=conf['model'], trainer=trainer)

    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and adapt ASR model with specified manifest files.")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to the training data folder")
    parser.add_argument("--adapt_data_path", type=str, required=True, help="Path to the adaptation data folder")
    
    args = parser.parse_args()
    
    main(args.train_data_path, args.adapt_data_path)
