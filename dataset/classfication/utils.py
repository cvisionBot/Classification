import cv2
import torch
import numpy as np

def visualize(images, classes, batch_idx=0):
    '''
    batch data visualize
    if you use Linux OS(Docker):
        save image file in docs dir and check
    '''
    img = images[batch_idx].numpy()
    img = (np.transpose(img, (1, 2, 0)) * 255.).astype(np.uint8).copy()
    label = classes[batch_idx].numpy()

    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    cv2.imwrite('/home/torch/Classification/docs/class'+str(label)+'.JPEG', img)

