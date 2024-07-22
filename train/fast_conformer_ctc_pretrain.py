from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import logging
import nemo.collections.asr as nemo_asr
from ruamel.yaml import YAML

config_path = 'configs/pretrain-ctc.yaml'

yaml = YAML(typ='safe')
with open(config_path) as f:
    params = yaml.load(f)
print(params)

params['model'].pop('test_ds')
logging.getLogger('nemo_logger').setLevel(logging.ERROR)

wandb_logger = WandbLogger(project="AIC-ASR", name="000_CTC_Pretraining")

trainer = pl.Trainer(logger=wandb_logger, max_epochs=500) # We stop at 15 epochs only

conf = OmegaConf.create(params)

model = nemo_asr.models.EncDecCTCModelBPE(cfg=conf['model'], trainer=trainer)

print(model)

trainer.fit(model)