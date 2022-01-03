# Lib Load
import os
import cv2
import glob
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

class TinyImageNetDataset(Dataset):
    def __init__(self, path, transforms, is_train):
        super(TinyImageNetDataset, self).__init__()
        self.transforms = transforms
        self.is_train = is_train
        with open(path + '/wnids.txt', 'r') as f:
            self.label_list = f.read().splitlines()

        if is_train:
            self.data = glob.glob(path + '/train/*/images/*.JPEG')
            self.train_list = dict()
            for data in self.data:
                label = data.split(os.sep)[-3]
                self.train_list[data] = self.label_list.index(label)

        else:
            self.data = glob.glob(path + '/val/images/*.JPEG')
            self.val_list = dict()
            with open(path + '/val/val_annotations.txt', 'r') as f:
                val_labels = f.read().splitlines()
                for label in val_labels:
                    f_name, label, _, _, _, _ = label.split('\t')
                    self.val_list[f_name] = self.label_list.index(label)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file = self.data[index]
        img = cv2.imread(img_file)
        if self.is_train:
            label = self.train_list[img_file]
        else:
            label = self.val_list[os.path.basename(img_file)]
        transformed = self.transforms(image=img)['image']
        # return {'img' : torch.stack([transformed]), 'class' : label}
        return transformed, label

class TinyImageNet(pl.LightningDataModule):
    def __init__(self, path, workers, train_transforms, val_transforms, batch_size=None):
        super(TinyImageNet, self).__init__()
        self.path = path
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.batch_size = batch_size
        self.workers = workers
    
    def train_dataloader(self):
        return DataLoader(TinyImageNetDataset(self.path,
                                              transforms=self.train_transforms,
                                              is_train=True),
                          batch_size=self.batch_size,
                          num_workers=self.workers,
                          persistent_workers=self.workers > 0,
                          pin_memory=self.workers > 0)

    def val_dataloader(self):
        return DataLoader(TinyImageNetDataset(self.path,
                                              transforms=self.val_transforms,
                                              is_train=False),
                          batch_size=self.batch_size,
                          num_workers=self.workers,
                          persistent_workers=self.workers > 0,
                          pin_memory=self.workers > 0)


if __name__ == '__main__':
    '''
    Dataset Loader Test
    run$ python -m dataset.classification/tiny_imagenet
    '''

    import albumentations
    import albumentations.pytorch
    from dataset.classification.utils import visualize

    train_transforms = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.ColorJitter(),
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2()])

    loader = DataLoader(TinyImageNetDataset(path='/ssd2/lyj/dataset/tiny-imagenet-200', transforms=train_transforms, is_train=True))
                            # batch_size=1, shuffle=True))
    for batch, sample in enumerate(loader):
        print('image : ', sample['img'])
        print('class : ', sample['class'])
        visualize(sample['img'], sample['class'])
        break
