import os
import numpy as np
import scipy.io as sio

import torch
from torch.utils.data import DataLoader, TensorDataset

__all__ = ['Cost2100DataLoader', 'MetaDataLoader']

class Cost2100DataLoader(object):
    r""" PyTorch DataLoader for COST2100 dataset.
    """

    def __init__(self, root, batch_size, num_workers, scenario, device):
        print(root)
        print(scenario)
        assert os.path.isdir(root)
        self.batch_size = batch_size
        self.batch_tiny_size = batch_size // 5
        self.num_workers = num_workers
        self.pin_memory = False
    
        dir_train = os.path.join(root, f"DATA_Htrain{scenario}.mat")
        dir_test = os.path.join(root, f"DATA_Htest{scenario}.mat")
        channel, nt, nc, nc_expand = 2, 32, 32, 125

        # Training data loading
        data_train = sio.loadmat(dir_train)['HT']
        data_train = torch.tensor(data_train, dtype=torch.float32).view(
            data_train.shape[0], channel, nt, nc).to(device)
        self.train_dataset = TensorDataset(data_train)
        self.train_tiny_dataset = TensorDataset(data_train[0:data_train.shape[0] // 5, :, :, :])
        
        data_test = sio.loadmat(dir_test)['HT']
        data_test = torch.tensor(data_test, dtype=torch.float32).view(
            data_test.shape[0], channel, nt, nc).to(device)

        self.test_dataset = TensorDataset(data_test)

    def __call__(self):
        train_loader = DataLoader(self.train_dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  pin_memory=self.pin_memory,
                                  shuffle=True)
        
        train_tiny_loader = DataLoader(self.train_tiny_dataset,
                                  batch_size=self.batch_tiny_size,
                                  num_workers=self.num_workers,
                                  pin_memory=self.pin_memory,
                                  shuffle=True)
        
        test_loader = DataLoader(self.test_dataset,
                                 batch_size=self.batch_size,
                                 num_workers=self.num_workers,
                                 pin_memory=self.pin_memory,
                                 shuffle=False)
        

        return train_loader, train_tiny_loader, test_loader
  
