from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import logging
import torch
import nemo.collections.asr as nemo_asr
from ruamel.yaml import YAML

config_path = 'configs/train-ctc.yaml'

yaml = YAML(typ='safe')
with open(config_path) as f:
    params = yaml.load(f)
print(params)

params['model'].pop('test_ds')

logging.getLogger('nemo_logger').setLevel(logging.ERROR)

wandb_logger = WandbLogger(project="AIC-ASR", name="000_CTC_Training")

trainer = pl.Trainer(logger=wandb_logger, max_epochs=20)

conf = OmegaConf.create(params)

model = nemo_asr.models.EncDecCTCModelBPE(cfg=conf['model'], trainer=trainer)
ckpt_path = "/kaggle/input/000-pretrain-ctc/AIC-ASR/i6we90fl/checkpoints/epoch=15-step=15632.ckpt"
model.load_state_dict(torch.load(ckpt_path)['state_dict'])
print(model)

trainer.fit(model)