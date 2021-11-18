import torch
from torch.utils.data import Dataset, DataLoader 
import numpy as np 

class CustomDataset(Dataset):

    def __init__(self, transform=None):

        self.x = np.random.random(size=(1024, 32, 32))
        self.y = np.random.random(size=(1024, 1))
        self.transform = transform 

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {"x": self.x[idx,:,:], "y":self.y[idx,:]}
        sample = (self.x[idx,:,:], self.y[idx,:])
        if self.transform:
            sample = self.transform(sample)
        return sample
 

ds = CustomDataset()
dl = DataLoader(ds, batch_size=32)
di = iter(dl)

print(di.next())
print(type(di.next()))
