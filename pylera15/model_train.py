import torch
import warnings

import pytorch_lightning as pl

from pyt_model import Transformerpyl
from config import get_config
cfg = get_config()

if __name__ == '__main__':
  warnings.filterwarnings("ignore")
  torch.cuda.empty_cache()
  pl.seed_everything(42, workers=True) #inconjuction with deterministci
  transformer_model = Transformerpyl(num_examples=2)
  trainer = pl.Trainer(
      max_epochs=10,
      precision=16,
      deterministic=True #deterministic for reproducibility
  )
  trainer.fit(transformer_model)
