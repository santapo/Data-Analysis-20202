import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from torch import optim
import torch.nn as nn

from extractor import get_autoencoder
from data import get_datamodule


class AutoEncoderModel(pl.LightningModule):
    def __init__(self,
                optimizer_name: str = "adam",
                lr_scheduler_method_name: str = 'none',
                lr: float = 0.001,
                momentum: float = 0.9,
                weight_decay: float = 5e-4,
                pretrained: bool = True,
                freeze_encoder: bool = False):
        """AutoEncoderModel
        Args:
        
        """
        super().__init__()
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name
        self.lr_scheduler_method_name = lr_scheduler_method_name
        self.train_loader_total_samples = 197
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
        
        if self.optimizer_name == "adam":
            optimizer = optim.Adam(trainable_parameters, lr=self.lr)
        elif self.optimizer_name == 'sgd':
            optimizer = optim.SGD(trainable_parameters, lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        elif self.optimizer_name == 'rmsprop':
            optimizer = optim.RMSprop(trainable_parameters, lr=self.lr)
        else:
            raise ValueError(f'Optimizer {self.optimizer_name} is not defined')
        optimizer_list = [optimizer]

        if self.lr_scheduler_method_name == 'none':
            scheduler_list = []
        else:
            if self.lr_scheduler_method_name == 'cosine':
                scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.train_loader_total_samples)
            elif self.lr_scheduler_method_name == 'cyclic':
                scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=self.lr, max_lr=self.max_lr)
            else:
                raise ValueError(f"Learning rate scheduler {self.lr_scheduler_method_name} are not defined")
            scheduler_list = [scheduler]
        
        return optimizer_list, scheduler_list

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

    cifar10_dm = get_datamodule(
        batch_size=256,
        num_workers=2
    )
    model = AutoEncoderModel(
        pretrained=True,
        optimizer_name='sgd',
        lr_scheduler_method_name='cosine',
        lr=0.005
    )
    model.datamodule = cifar10_dm
    model.load_from_checkpoint('exps/vgg16_sgd_256_cosine/lightning_logs/version_1/checkpoints/epoch=99-step=15699-val_loss=0.07.ckpt')
    save_checkpoint = ModelCheckpoint(dirpath=None,
                                    filename='{epoch}-{step}-{val_loss:.2f}',
                                    save_top_k=-1)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    
    trainer = pl.Trainer(
        default_root_dir='exps/vgg16_sgd_256_cosine',
        gpus=1,
        max_epochs=100,
        callbacks=[save_checkpoint, lr_monitor],
    )
    
    trainer.fit(model, cifar10_dm)

    

    