# -------------------------
# 2021.11.23 Bumjin Park 
# -------------------------

from implicit_learning.utils import compute_image_gradient, get_mgrid
import torch
from torch.utils.data import Dataset, DataLoader 
import scipy.ndimage
import copy
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize, Scale


# class CustomDataset(Dataset):
#     def __init__(self, transform):
#         self.x = np.random.random(size=(32*32, 1,2))
#         self.y = np.random.random(size=(32*32, 1,3))
#         self.transform = transform 

#     def __len__(self):
#         return self.x.shape[0]

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()

#         sample = (self.x[idx,:], self.y[idx,:])
#         if self.transform:
#             sample = self.transform(sample[0]).float(), self.transform(sample[1]).float()
#         return sample
 
# class SingleImageDataset(CustomDataset):
#     def __init__(self, config, type, transform):
#         im = Image.open(config['data-path'])
        
#         self.data = np.array(im)  #(width, height, channel)       
#         self.is_gray = False 
#         self.width  = self.data.shape[0]
#         self.height = self.data.shape[1]
        
#         self.flat_data = self.data.reshape(-1,1, self.data.shape[-1]) # row1 first. 
#         self.transform = transform 
#         self.device  = torch.device("cuda" if config.get("gpu", False) else 'cpu')

#     def __len__(self):
#         return self.flat_data.shape[0]

#     def generate_position_by_idx(self, idx):
#         pos_x = (idx%self.width ) / self.width
#         pos_y = (idx// self.height) / self.height
#         return np.array([[pos_x, pos_y]])

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         sample = [self.generate_position_by_idx(idx), self.flat_data[idx]]
        
#         if self.transform:
#             sample = [self.transform(sample[0]).float(), self.transform(sample[1]).float()]
    
#         sample[1]/=255
#         if torch.cuda.is_available():
#             sample[0] = sample[0].to(device=self.device)
#             sample[1] = sample[1].to(device=self.device)

#         return sample
 


# class PoissonEqn(Dataset):
#     def __init__(self, config, transform, scaler):
#         super().__init__()
#         img = Image.open(config['data-path'])
#         img = transform(img)
#         img = img.mean(axis=0).unsqueeze(0)
#         img = scaler.fit_transform(img)

#         sidelength = config.get("sidelength")
        
#         # --- Compute gradient and laplacian       
#         RGB_grads = [] 
#         RGB_pixels = [] 
#         RGB_laplace = [] 
#         for i in range(img.shape[0]): 
#             temp = copy.deepcopy(img[i,:,:])
#             temp = temp.unsqueeze(0).numpy()
#             grads_x, grads_y = compute_image_gradient(temp, type=config.get("gradient_type"))
#             grads_x, grads_y = torch.from_numpy(grads_x), torch.from_numpy(grads_y)

#             grads = torch.stack((grads_x, grads_y), dim=-1).view(-1,2)
#             pixels = img[i,:,:].unsqueeze(0).permute(1, 2, 0).view(-1, 1)
#             laplace = scipy.ndimage.laplace(temp).squeeze(0)[..., None]
#             laplace = torch.from_numpy(laplace)       
#             RGB_pixels.append(pixels)     
#             RGB_grads.append(grads)     
#             RGB_laplace.append(laplace)

#         self.coords  = get_mgrid(sidelength, 2)
#         self.pixels  = torch.stack(RGB_pixels,dim=1).view(-1,  img.shape[0])
#         self.grads   = torch.stack(RGB_grads,dim=1).view(-1, img.shape[0], 2)
#         self.laplace = torch.stack(RGB_laplace).view(sidelength, sidelength,1 ,img.shape[0])
#     def __len__(self):
#         return 1

#     def __getitem__(self, idx):
#         return self.coords, {'pixels':self.pixels, 'grads':self.grads, 'laplace':self.laplace}


import numpy as np 
class ImageDataset(Dataset):
    def __init__(self, config,scaler):
        super().__init__()
        img = Image.open(config['data-path'])

        p = Compose([Scale((256,256))])        
        transform = Compose([
            Resize(256),
            ToTensor(),
            #Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
                ])
        img = p(img)
        img = transform(img)
        if len(img.size()) ==2:
            img = img.unsqueeze(0)
        if config['model']['out_features'] ==1 and img.size(0) !=1:
            img = img.mean(axis=0).unsqueeze(0)
        elif config['model']['out_features'] != img.size(0):
            raise ValueError()
        

        img = scaler.fit_transform(img)
        sidelength_W = img.size(1)
        sidelength_H = img.size(2)
        
        # --- Compute gradient and laplacian       
        RGB_grads = [] 
        RGB_pixels = [] 
        RGB_laplace = [] 
        for i in range(img.shape[0]): 
            temp = copy.deepcopy(img[i,:,:])
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

        self.coords  = get_mgrid(sidelength_W, sidelength_H, 2)
        self.pixels  = torch.stack(RGB_pixels,dim=1).view(-1,  img.shape[0])
        self.grads   = torch.stack(RGB_grads,dim=1).view(-1, img.shape[0], 2)
        self.laplace = torch.stack(RGB_laplace).view(sidelength_W, sidelength_H,1 ,img.shape[0])
        self.side_lengths = (sidelength_W, sidelength_H)
        print(self.coords.size())

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.coords, {'pixels':self.pixels, 'grads':self.grads, 'laplace':self.laplace}


if __name__ == "__main__":
    ds = ImageDataset()
    dl = DataLoader(ds, batch_size=32)
    di = iter(dl)
    print(di.next())
    print(type(di.next()))
