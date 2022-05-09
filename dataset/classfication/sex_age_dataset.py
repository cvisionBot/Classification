import os

from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import albumentations
import albumentations.pytorch
import cv2
import numpy as np


class SexAgeDataset(Dataset):
    def __init__(self, file_path, transforms):
        super().__init__()
        self.transforms = transforms

        with open(file_path, 'r') as f:
            self.data = f.read().splitlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file = self.data[index]
        img = cv2.imread(img_file)
        transformed = self.transforms(image=img)['image']

        sex, age, _ = os.path.basename(img_file).split('_')

        label = []
        if sex == 'male':
            label.append(0)
        elif sex == 'female':
            label.append(1)

        if age == '10':
            label.append(0)
        elif age == '20':
            label.append(1)
        elif age == '30':
            label.append(2)
        elif age == '40':
            label.append(3)
        elif age == '50':
            label.append(4)
        elif age == '60':
            label.append(5)

        return transformed, label


class SexAge(pl.LightningDataModule):
    def __init__(self, dataset_dir, workers, batch_size, input_size):
        super().__init__()
        self.train_path = os.path.join(dataset_dir, 'train.txt')
        self.val_path = os.path.join(dataset_dir, 'val.txt')
        self.workers = workers
        self.batch_size = batch_size
        self.input_size = input_size
        
    def setup(self, stage=None):
        train_transforms = albumentations.Compose([
            albumentations.HorizontalFlip(),
            albumentations.Blur(),
            albumentations.GaussNoise(),
            albumentations.Cutout(max_h_size=int(self.input_size*0.2), max_w_size=int(self.input_size*0.2)),
            albumentations.Resize(self.input_size*2, self.input_size, always_apply=True),
            albumentations.Normalize(0, 1),
            albumentations.pytorch.ToTensorV2(),
        ],)

        valid_transform = albumentations.Compose([
            albumentations.Resize(self.input_size*2, self.input_size, always_apply=True),
            albumentations.Normalize(0, 1),
            albumentations.pytorch.ToTensorV2(),
        ],)

        self.train_dataset = SexAgeDataset(
            self.train_path,
            train_transforms
        )

        self.valid_dataset = SexAgeDataset(
            self.val_path,
            valid_transform
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
            pin_memory=self.workers > 0
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
            pin_memory=self.workers > 0
        )


if __name__ == '__main__':
    input_size = 224
    train_path = '/home/fssv2/myungsang/datasets/aihub_human_action/sex_age_dataset/train.txt'
    val_path = '/home/fssv2/myungsang/datasets/aihub_human_action/sex_age_dataset/val.txt'

    test_transform = albumentations.Compose([
        albumentations.HorizontalFlip(),
        albumentations.Blur(),
        # albumentations.ShiftScaleRotate(),
        albumentations.GaussNoise(),
        albumentations.Cutout(max_h_size=int(input_size*0.2), max_w_size=int(input_size*0.2)),
        # albumentations.ElasticTransform(),
        albumentations.Resize(input_size*2, input_size, always_apply=True),
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ],)
    
    test_dataset = SexAgeDataset(
        val_path,
        test_transform
    )

    data_loader = DataLoader(
        test_dataset,
        1,
        True
    )

    for sample in data_loader:
        batch_x, batch_y = sample

        img = batch_x[0].numpy()   
        img = (np.transpose(img, (1, 2, 0))*255.).astype(np.uint8).copy()

        print(batch_y)
        

        cv2.imshow('sample', img)
        key = cv2.waitKey(0)
        if key == 27:
            break

    cv2.destroyAllWindows()
