from pytorch_lightning import callbacks
from extractor.vgg16_autoencoder import AutoEncoder
from torch import optim
import torch.nn as nn
from extractor import get_autoencoder

import pytorch_lightning as pl


class AutoEncoderModel(pl.LightningModule):
    def __init__(self,
                optimizer_name: str = "Adam",
                lr: float = 0.001,
                pretrained: bool = True,
                freeze_encoder: bool = False):
        """AutoEncoderModel
        Args:
        
        """
        super().__init__()
        self.lr = lr
        self.optimizer_name = optimizer_name
        self.inp_freeze_encoder = freeze_encoder
        self.model = get_autoencoder(model_name='vgg16_bn', pretrained=pretrained)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        """
        Model Forward Path
        """
        return self.model(x)
    
    def configure_optimizers(self):
        """
        Configuration of Optimizer and LR Scheduler
        """
        if self.inp_freeze_encoder:
            trainable_parameters = self.model.decoder.parameters()
        else:
            trainable_parameters = self.model.parameters()
        
        if self.optimizer_name == "Adam":
            optimizer = optim.Adam(trainable_parameters, lr=self.lr)
        else:
            raise ValueError(f'Optimizer {self.optimizer_name} is not defined')
        
        return optimizer

    def training_step(self, batch, batch_idx):
        """
        Training step for each batch of an training iteration.
        """
        image, _ = batch
        out = self.forward(image)
        loss = self.criterion(out, image)
        self.log('train_loss', loss, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        Validation step for each batch of a validation iteration.
        """
        image, _ = batch
        out = self.forward(image)
        loss = self.criterion(out, image)
        self.log('val_loss', loss, on_step=True)


if __name__ == "__main__":
    from data import get_datamodule
    from pytorch_lightning.callbacks import ModelCheckpoint

    cifar10_dm = get_datamodule()
    model = AutoEncoderModel(pretrained=True)
    model.datamodule = cifar10_dm
    save_checkpoint = ModelCheckpoint(dirpath=None,
                                    filename='{epoch}-{step}-{val_loss:.2f}',
                                    save_top_k=-1)
    trainer = pl.Trainer(
        default_root_dir='exps',
        max_epochs=30,
        callbacks=[save_checkpoint]
    )
    
    trainer.fit(model, cifar10_dm)

    

    