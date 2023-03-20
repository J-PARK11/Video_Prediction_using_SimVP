import os
import numpy as np
import cv2
import torch
from torch.utils.data import TensorDataset
from utils import print_log


def load_data(root, batch_size, val_batch_size, num_workers, out_frame):
    path = os.path.join(root, 'caltech/')
    train_dataset = np.load(path + 'dataset4Train.npz')
    test_dataset = np.load(path + 'dataset4Test.npz')

    train_x = train_dataset['x'].swapaxes(2,4).swapaxes(3,4)
    train_y = train_dataset['y'].swapaxes(2,4).swapaxes(3,4)
    test_x = test_dataset['x'].swapaxes(2,4).swapaxes(3,4)
    test_y = test_dataset['y'].swapaxes(2,4).swapaxes(3,4)

    train_rand_idx = np.arange(train_x.shape[0])
    test_rand_idx = np.arange(test_x.shape[0])
    np.random.shuffle(train_rand_idx)
    np.random.shuffle(test_rand_idx)
    train_x, train_y = train_x[train_rand_idx], train_y[train_rand_idx]
    test_x, test_y = test_x[test_rand_idx], test_y[test_rand_idx]
    
    train_x, train_y = torch.FloatTensor(train_x), torch.FloatTensor(train_y)
    test_x, test_y = torch.FloatTensor(test_x), torch.FloatTensor(test_y)
    
    train_mean = torch.cat([train_x,train_y],dim=1).mean().item()
    train_std = torch.cat([train_x,train_y],dim=1).std().item()
    test_mean = torch.cat([test_x,test_y],dim=1).mean().item()
    test_std = torch.cat([test_x,test_y],dim=1).std().item()

    train_set = TensorDataset(train_x, train_y)
    test_set = TensorDataset(test_x, test_y)

    # make dataloader
    dataloader_train, dataloader_validation, dataloader_test, mean, std = make_dataloader(
        train_set, test_set, batch_size, val_batch_size, num_workers, out_frame
    )
    return dataloader_train, dataloader_validation, dataloader_test, train_mean, train_std, test_mean, test_std


def make_dataloader(train_set, test_set, batch_size, val_batch_size, num_workers, out_frame):
    
    dataloader_train = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, pin_memory=True, num_workers=num_workers)
    dataloader_validation = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, pin_memory=True, num_workers=num_workers)
    dataloader_test = torch.utils.data.DataLoader(
        test_set, batch_size=val_batch_size, pin_memory=True, num_workers=num_workers)
    
    mean, std = 0, 1
    return dataloader_train, dataloader_validation, dataloader_test, mean, std