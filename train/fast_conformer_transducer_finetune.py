from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import nemo.collections.asr as nemo_asr
from ruamel.yaml import YAML
import argparse
import torch

parser = argparse.ArgumentParser(description='Training Script')
parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to the model checkpoint')
args = parser.parse_args()

config_path = 'configs/adapt-transducer.yaml'

yaml = YAML(typ='safe')
with open(config_path) as f:
    params = yaml.load(f)
print(params)

params['model'].pop('test_ds')

wandb_logger = WandbLogger(project="AIC-ASR", name="005_FastConformerM_transd")

trainer = pl.Trainer(max_epochs=500, logger=wandb_logger, check_val_every_n_epoch=1, accumulate_grad_batches=32)

conf = OmegaConf.create(params)
print(OmegaConf.to_yaml(conf, resolve=True))

model = nemo_asr.models.EncDecRNNTBPEModel(cfg=conf['model'], trainer=trainer)

# Load the checkpoint if the path is provided
if args.checkpoint_path:
    ckpt = torch.load(args.checkpoint_path)
    model.load_state_dict(ckpt['state_dict'])
    print(f"Loaded model state from {args.checkpoint_path}")

print(model)

trainer.fit(model)
