import argparse
import platform

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging, QuantizationAwareTraining
from pytorch_lightning.plugins import DDPPlugin
import torchsummary

from module.sex_age_classifier import SexAgeClassifier
from models.classifier.sex_age import SexAge
from utils.utility import make_model_name
from utils.module_select import get_model, get_data_module
from utils.yaml_helper import get_configs


def add_experimental_callbacks(cfg, train_callbacks):
    options = {
        'SWA': StochasticWeightAveraging(),
        'QAT': QuantizationAwareTraining()
    }
    callbacks = cfg['experimental_options']['callbacks']
    if callbacks:
        for option in callbacks:
            train_callbacks.append(options[option])

    return train_callbacks


def train(cfg):
    data_module = get_data_module(cfg['dataset_name'])(
        dataset_dir=cfg['dataset_dir'],
        workers=cfg['workers'],
        batch_size=cfg['batch_size'],
        input_size=cfg['input_size']
    )

    model = SexAge(
        backbone=get_model(cfg['model']),
        in_channels=cfg['in_channels'],
        input_size=cfg['input_size']
    )

    torchsummary.summary(model, (cfg['in_channels'], cfg['input_size']*2, cfg['input_size']), batch_size=1, device='cpu')
    
    model_module = SexAgeClassifier(
        model=model,
        cfg=cfg
    )

    callbacks = [
        LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(
            monitor='val_loss', 
            save_last=True, 
            every_n_epochs=cfg['save_freq']
        )
    ]

    # callbacks = add_expersimental_callbacks(cfg, callbacks)

    trainer = pl.Trainer(
        max_epochs=cfg['epochs'],
        logger=TensorBoardLogger(cfg['save_dir'], make_model_name(cfg), default_hp_metric=False),
        accelerator=cfg['accelerator'],
        devices=cfg['devices'],
        plugins=DDPPlugin(find_unused_parameters=False) if platform.system() != 'Windows' else None,
        callbacks=callbacks,
        **cfg['trainer_options']
    )

    trainer.fit(model_module, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='config file')
    args = parser.parse_args()
    cfg = get_configs(args.cfg)

    train(cfg)
