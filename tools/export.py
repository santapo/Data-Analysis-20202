import os
import sys

import torch

from train_autoencoder import AutoEncoderModel

model = AutoEncoderModel.load_from_checkpoint('exps/logs/epoch=0-step=1249-val_loss=0.27.ckpt')
del model.criterion
torch.save(model.state_dict(), 'model.pth')
