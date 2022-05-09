import pytorch_lightning as pl
import torch.nn.functional as F
from torchmetrics import Accuracy

from utils.module_select import get_optimizer, get_scheduler


class SexAgeClassifier(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.model = model
        self.save_hyperparameters(ignore='model')
        self.top_1 = Accuracy(top_k=1)

    def forward(self, x):
        predictions = self.model(x)
        return predictions

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred_sex, pred_age = self.model(x)
        sex_loss = F.cross_entropy(pred_sex, y[0])
        age_loss = F.cross_entropy(pred_age, y[1])
        loss = sex_loss + age_loss

        self.log('train_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred_sex, pred_age = self.model(x)
        sex_loss = F.cross_entropy(pred_sex, y[0])
        age_loss = F.cross_entropy(pred_age, y[1])
        loss = sex_loss + age_loss
        
        self.log('val_loss', loss, prog_bar=True, logger=True, on_epoch=True, on_step=False)
        self.log('val_sex_accuracy', self.top_1(pred_sex, y[0]), logger=True, on_epoch=True, on_step=False)
        self.log('val_age_accuracy', self.top_1(pred_age, y[1]), logger=True, on_epoch=True, on_step=False)
    
    def configure_optimizers(self):
        cfg = self.hparams.cfg
        optim = get_optimizer(
            cfg['optimizer'],
            self.model.parameters(),
            **cfg['optimizer_options']
        )
        
        try:
            scheduler = get_scheduler(
                cfg['scheduler'],
                optim,
                **cfg['scheduler_options']
            )
    
            return {
                "optimizer": optim,
                "lr_scheduler": {
                    "scheduler": scheduler
                }
            } 
        
        except KeyError:
            return optim
