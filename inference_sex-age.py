import argparse
import time

import numpy as np
import cv2
import torch

from models.classifier.sex_age import SexAge
from module.sex_age_classifier import SexAgeClassifier
from utils.module_select import get_model, get_data_module
from utils.yaml_helper import get_configs


def inference(cfg, ckpt_path):
    data_module = get_data_module(cfg['dataset_name'])(
        dataset_dir=cfg['dataset_dir'],
        workers=cfg['workers'],
        batch_size=1,
        input_size=cfg['input_size']
    )
    data_module.prepare_data()
    data_module.setup()

    model = SexAge(
        backbone=get_model(cfg['model']),
        in_channels=cfg['in_channels'],
        input_size=cfg['input_size']
    )

    if torch.cuda.is_available:
        model = model.to('cuda')

    model_module = SexAgeClassifier.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        model=model,
        cfg=cfg
    )
    model_module.eval()

    # Inference
    for sample in data_module.val_dataloader():
        batch_x, batch_y = sample

        if torch.cuda.is_available:
            batch_x = batch_x.cuda()    
        
        before = time.time()
        with torch.no_grad():
            pred_sex, pred_age = model_module(batch_x)
        print(f'Inference: {(time.time()-before)*1000:.2f}ms')

        if batch_y[0] == 0:
            true_sex = '남성'
        else: 
            true_sex = '여성'
        
        if batch_y[1] == 0:
            true_age = '10대'
        elif batch_y[1] == 1:
            true_age = '20대'
        elif batch_y[1] == 2:
            true_age = '30대'
        elif batch_y[1] == 3:
            true_age = '40대'
        elif batch_y[1] == 4:
            true_age = '50대'
        elif batch_y[1] == 5:
            true_age = '60대'

        pred_sex_idx = torch.argmax(pred_sex)
        if pred_sex_idx == 0:
            pred_sex_str = '남성'
        else: 
            pred_sex_str = '여성'
        
        pred_age_idx = torch.argmax(pred_age)
        if pred_age_idx == 0:
            pred_age_str = '10대'
        elif pred_age_idx == 1:
            pred_age_str = '20대'
        elif pred_age_idx == 2:
            pred_age_str = '30대'
        elif pred_age_idx == 3:
            pred_age_str = '40대'
        elif pred_age_idx == 4:
            pred_age_str = '50대'
        elif pred_age_idx == 5:
            pred_age_str = '60대'

        print(f'Label: {true_sex}, {true_age}, Prediction: {pred_sex_str}, {pred_age_str}')

        # batch_x to img
        if torch.cuda.is_available:
            img = batch_x.cpu()[0].numpy()   
        else:
            img = batch_x[0].numpy()   
        img = (np.transpose(img, (1, 2, 0))*255.).astype(np.uint8).copy()

        cv2.imshow('Result', img)
        key = cv2.waitKey(0)
        if key == 27:
            break
    
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str, help='config file')
    parser.add_argument('--ckpt', required=True, type=str, help='ckpt file path')
    args = parser.parse_args()
    cfg = get_configs(args.cfg)

    inference(cfg, args.ckpt)
