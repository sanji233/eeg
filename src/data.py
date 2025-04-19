import numpy as np
import torch
from torch.utils.data import Dataset

class EEGDataset(Dataset):
    """
    split the dataset -> train,val,test

    """
    def __init__(self, x, y=None, split=None):
        # x: (n_samples, n_channels, n_times)
        # y: (n_samples, )
        super().__init__()
        self.__split = None

        N_SAMPLE = x.shape[0]
        val_idx = int(0.9 * N_SAMPLE)
        train_idx = int(0.81 * N_SAMPLE)

        if y is not None:
            self.train_ds = {
                'x': x[:train_idx],
                'y': y[:train_idx],
            }
            self.val_ds = {
                'x': x[train_idx:val_idx],
                'y': y[train_idx:val_idx],
            }
            self.test_ds = {
                'x': x[val_idx:],
                'y': y[val_idx:],
            }
        else:
            self.inference_ds = {'x': x}

    def __len__(self):
        return len(self.dataset['x'])

    def __getitem__(self, idx):
        x_ = torch.tensor(self.dataset['x'][idx], dtype=torch.float32)  # shape=(n_channels, n_times)
        if self.__split != "inference":
            y_ = torch.tensor(self.dataset['y'][idx], dtype=torch.float32).unsqueeze(-1)  # shape=(1,)
            return x_, y_
        else:
            return x_

    def split(self, __split):
        self.__split = __split
        return self

    @property
    def dataset(self):
        assert self.__split is not None, "Specify the split!"
        if self.__split == "train":
            return self.train_ds
        elif self.__split == "val":
            return self.val_ds
        elif self.__split == "test":
            return self.test_ds
        elif self.__split == "inference":
            return self.inference_ds
        else:
            raise TypeError("Unknown dataset split!")
    