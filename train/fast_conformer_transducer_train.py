"""
Example Usage:
python fast_conformer_transducer_train.py --checkpoint_path "/path/to/your/checkpoint.ckpt"
"""
import argparse
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import torch
import nemo.collections.asr as nemo_asr
from ruamel.yaml import YAML

parser = argparse.ArgumentParser(description='Train a Fast Conformer Transducer model with a specified checkpoint.')
parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint.')
args = parser.parse_args()

config_path = 'configs/train-transducer.yaml'

yaml = YAML(typ='safe')
with open(config_path) as f:
    params = yaml.load(f)
print(params)

params['model'].pop('test_ds')

wandb_logger = WandbLogger(project="AIC-ASR", name="005_FastConformerM_transd")

trainer = pl.Trainer(max_epochs=500, logger=wandb_logger, check_val_every_n_epoch=1) # We stop at ~85 epochs only

conf = OmegaConf.create(params)
print(OmegaConf.to_yaml(conf, resolve=True))

model = nemo_asr.models.EncDecRNNTBPEModel(cfg=conf['model'], trainer=trainer)

ckpt_path = args.checkpoint_path
ckpt = torch.load(ckpt_path)

ckpt_2 = dict()
for k, v in ckpt['state_dict'].items():
    if k.startswith('encoder'):
        ckpt_2[k[8:]] = v
        
model.encoder.load_state_dict(ckpt_2)

print(model)

trainer.fit(model)