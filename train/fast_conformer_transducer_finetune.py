from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import nemo.collections.asr as nemo_asr
from ruamel.yaml import YAML

config_path = 'FC-transducer.yaml'

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
print(model)

trainer.fit(model)