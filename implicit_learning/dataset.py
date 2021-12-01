# -------------------------
# 2021.11.23 Bumjin Park 
# -------------------------

from implicit_learning.utils import compute_image_gradient
import torch
from torch.utils.data import Dataset, DataLoader 
import numpy as np 
import scipy.ndimage


class CustomDataset(Dataset):
    def __init__(self, transform):
        self.x = np.random.random(size=(32*32, 1,2))
        self.y = np.random.random(size=(32*32, 1,3))
        self.transform = transform 

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = (self.x[idx,:], self.y[idx,:])
        if self.transform:
            sample = self.transform(sample[0]).float(), self.transform(sample[1]).float()
        return sample
 

from PIL import Image

class SingleImageDataset(CustomDataset):
    def __init__(self, config, type, transform):
        im = Image.open(config['data-path'])
        
        self.data = np.array(im)  #(width, height, channel)       
        self.is_gray = False 
        if len(self.data.shape) ==2:
            self.is_gray = True 
        self.width  = self.data.shape[0]
        self.height = self.data.shape[1]
        
        self.flat_data = self.data.reshape(-1,1, self.data.shape[-1]) # row1 first. 
        self.transform = transform 
        self.device  = torch.device("cuda" if config.get("gpu", False) else 'cpu')

    def __len__(self):
        return self.flat_data.shape[0]

    def generate_position_by_idx(self, idx):
        pos_x = (idx%self.width ) / self.width
        pos_y = (idx// self.height) / self.height
        return np.array([[pos_x, pos_y]])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = [self.generate_position_by_idx(idx), self.flat_data[idx]]
        
        if self.transform:
            sample = [self.transform(sample[0]).float(), self.transform(sample[1]).float()]
    
        sample[1]/=255
        if torch.cuda.is_available():
            sample[0] = sample[0].to(device=self.device)
            sample[1] = sample[1].to(device=self.device)

        return sample
 
def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

class PoissonEqn(Dataset):
    def __init__(self, config, transform, scaler):
        super().__init__()
        img = Image.open(config['data-path'])
        img = transform(img)
        img = img.mean(axis=0)
        img = scaler.fit_transform(img)

        img = img.unsqueeze(0)
        sidelength = config.get("sidelength")
        # Compute gradient and laplacian       
        grads_x, grads_y = compute_image_gradient(img.numpy(), type=config.get("gradient_type"))
        grads_x, grads_y = torch.from_numpy(grads_x), torch.from_numpy(grads_y)
                
        self.grads = torch.stack((grads_x, grads_y), dim=-1).view(-1,1, 2)
        self.laplace = scipy.ndimage.laplace(img.numpy()).squeeze(0)[..., None]
        self.laplace = torch.from_numpy(self.laplace)
        
        self.pixels = img.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.coords, {'pixels':self.pixels, 'grads':self.grads, 'laplace':self.laplace}


class PoissonEqnRGB(Dataset):
    def __init__(self, config, transform,scaler):
        super().__init__()
        img = Image.open(config['data-path'])
        img = transform(img)
        img = scaler.fit_transform(img)

        sidelength = config.get("sidelength")
        
        # --- Compute gradient and laplacian       
        RGB_grads = [] 
        RGB_pixels = [] 
        RGB_laplace = [] 
        for i in range(img.shape[0]):
            temp = img[i,:,:]
            temp = temp.unsqueeze(0).numpy()
            grads_x, grads_y = compute_image_gradient(temp, type=config.get("gradient_type"))
            grads_x, grads_y = torch.from_numpy(grads_x), torch.from_numpy(grads_y)

            grads = torch.stack((grads_x, grads_y), dim=-1).view(-1,2)
            pixels = img[i,:,:].unsqueeze(0).permute(1, 2, 0).view(-1, 1)
            laplace = scipy.ndimage.laplace(temp).squeeze(0)[..., None]
            laplace = torch.from_numpy(laplace)       
            RGB_pixels.append(pixels)     
            RGB_grads.append(grads)     
            RGB_laplace.append(laplace)

        self.coords  = get_mgrid(sidelength, 2)
        self.pixels  = torch.stack(RGB_pixels,dim=2).view(-1,  img.shape[0])
        self.grads   = torch.stack(RGB_grads,dim=2).view(-1, img.shape[0], 2)
        self.laplace = torch.stack(RGB_laplace).view(sidelength, sidelength,1 ,img.shape[0])

        
    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.coords, {'pixels':self.pixels, 'grads':self.grads, 'laplace':self.laplace}



if __name__ == "__main__":
    ds = CustomDataset()
    dl = DataLoader(ds, batch_size=32)
    di = iter(dl)
    print(di.next())
    print(type(di.next()))
