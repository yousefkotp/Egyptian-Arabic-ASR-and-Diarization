from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import logging
import torch
import nemo.collections.asr as nemo_asr
from ruamel.yaml import YAML
import argparse

# Define argument parser
parser = argparse.ArgumentParser(description='Training Script')
parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to the model checkpoint')
args = parser.parse_args()

config_path = 'configs/train-ctc.yaml'

yaml = YAML(typ='safe')
with open(config_path) as f:
    params = yaml.load(f)
print(params)

params['model'].pop('test_ds')

logging.getLogger('nemo_logger').setLevel(logging.ERROR)

wandb_logger = WandbLogger(project="AIC-ASR", name="000_CTC_Training")

trainer = pl.Trainer(logger=wandb_logger, max_epochs=500) # We stop at 20 epochs only

conf = OmegaConf.create(params)

model = nemo_asr.models.EncDecCTCModelBPE(cfg=conf['model'], trainer=trainer)

if args.checkpoint_path:
    ckpt = torch.load(args.checkpoint_path)
    model.load_state_dict(ckpt['state_dict'])
    print(f"Loaded model state from {args.checkpoint_path}")

print(model)

trainer.fit(model)