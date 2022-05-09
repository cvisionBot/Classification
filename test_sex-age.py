import argparse
import platform

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
import torchsummary

from models.classifier.sex_age import SexAge
from module.sex_age_classifier import SexAgeClassifier
from utils.module_select import get_model, get_data_module
from utils.yaml_helper import get_configs


def test(cfg, ckpt_path):
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

    model_module = SexAgeClassifier.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        model=model,
        cfg=cfg
    )

    trainer = pl.Trainer(
        logger=False,
        accelerator=cfg['accelerator'],
        devices=cfg['devices'],
        plugins=DDPPlugin(find_unused_parameters=False) if platform.system() != 'Windows' else None,
    )

    trainer.validate(model_module, data_module)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='config file')
    parser.add_argument('--ckpt', required=True, type=str, help='ckpt file path')
    args = parser.parse_args()
    cfg = get_configs(args.cfg)

    test(cfg, args.ckpt)
