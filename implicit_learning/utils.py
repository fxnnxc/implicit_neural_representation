# -------------------------
# 2021.11.23 Bumjin Park 
# -------------------------

import numpy as np 
import scipy 
import torch 
import math
from skimage.metrics import structural_similarity as ssim


def compute_image_gradient(im_array, type, v=1):
    LENGTH = im_array.shape[1], im_array.shape[2]
    im_array = im_array.reshape(*LENGTH)
    magnitude = 1
    
    if type =="sobel":
        grads_x = scipy.ndimage.sobel(im_array, axis=0)[..., None]
        grads_y = scipy.ndimage.sobel(im_array, axis=1)[..., None]
        magnitude = (2*2 + 1*4)

    elif type=="sobel_same": # same sobel result  
        grads_x = scipy.ndimage.correlate1d(im_array, [-1,0,1], axis=0)   # x-axis
        grads_x = scipy.ndimage.correlate1d(grads_x,  [ 1,2,1], axis=1)[..., None]   # y-axis
        grads_y = scipy.ndimage.correlate1d(im_array, [-1,0,1], axis=1)   # y-axis
        grads_y = scipy.ndimage.correlate1d(grads_y,  [ 1,2,1], axis=0)[..., None]  # x-axis
        magnitude = (2*2 + 1*4)
    
    elif type=="bumjin": # same sobel result  
        grads_x = scipy.ndimage.correlate1d(im_array, [-v,0,v], axis=0)
        grads_x = scipy.ndimage.correlate1d(grads_x, [2*v*np.sqrt(2), v*2, 2*v*np.sqrt(2)], axis=1)[..., None]
        grads_y = scipy.ndimage.correlate1d(im_array, [-v,0,v], axis=1)
        grads_y = scipy.ndimage.correlate1d(grads_y, [2*v*np.sqrt(2), v*2, 2*v*np.sqrt(2)], axis=0)[..., None]
        magnitude = (2*(2*v) + 2*v*np.sqrt(2)*4)

    elif type=="wonjoon": # same sobel result  
        grads_x = scipy.ndimage.correlate1d(im_array, [-v,0,v], axis=1)
        grads_x = scipy.ndimage.correlate1d(grads_x, [v/np.sqrt(2), v, v/np.sqrt(2)], axis=0)[..., None]

        grads_y = scipy.ndimage.correlate1d(im_array, [-v,0,v], axis=0)
        grads_y = scipy.ndimage.correlate1d(grads_y, [v/np.sqrt(2), v, v/np.sqrt(2)], axis=1)[..., None]

    elif type=="convolve":  
        grads_x = scipy.ndimage.correlate1d(im_array, [-1,0,1], axis=0)[..., None]
        grads_y = scipy.ndimage.correlate1d(im_array, [-1,0,1], axis=1)[..., None]
        magnitude = 2

    elif type=="test": # same sobel result  
            grads_x = scipy.ndimage.correlate1d(im_array, [-1,0,1], axis=0)
            grads_x = scipy.ndimage.correlate1d(grads_x, [4,1 ,4], axis=1)[..., None]
            grads_y = scipy.ndimage.correlate1d(im_array, [-1,0,1], axis=1)
            grads_y = scipy.ndimage.correlate1d(grads_y, [1,2,1], axis=0)[..., None]
            magnitude = 18, 8
    else:
        raise ValueError("Not implemented gradient type")

    grads_x /= (magnitude/LENGTH[0])
    grads_y /= (magnitude/LENGTH[1])
    grads_x = grads_x.reshape(*LENGTH, 1)
    grads_y = grads_y.reshape(*LENGTH, 1)
    
    return grads_x, grads_y

def compute_image_divergence(img_array):
    x = scipy.ndimage.laplace(img_array)
    # x = np.sum(np.gradient(img_array),axis=0)
    return x 

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)

def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calculate_ssim(img1, img2):
    return ssim(img1, img2, data_range=max(img1.max(), img2.max()) - min(img1.min(), img2.min()), multichannel=True)

class MinMaxScaler():
    def __init__(self, channel_dim):
        self.channel_dim = channel_dim
        self.MAX = 0 
        self.MIN = 0
        self.channel_statistics = []

    def fit_transform(self, img):
        if len(img.size()) >2:
            for c in range(self.channel_dim):
                MAX = img[c,:,:].max()
                MIN = img[c,:,:].min()
                self.channel_statistics.append((MAX.item(), MIN.item()))
                if MAX > MIN:
                    img[c,:,:] = (img[c,:,:] - MIN)/(MAX-MIN) -0.5
                else:
                    img[c,:,:] = (img[c,:,:] - MIN) -0.5
            return img
        elif len(img.size()) ==2:
            self.MAX = img.max()
            self.MIN = img.min()
            assert self.MAX > self.MIN
            return (img - self.MIN)/(self.MAX-self.MIN)
        else:
            raise NotImplementedError()
    
    def inverse_transform(self, img):
        if len(img.shape) >2:
            for c in range(self.channel_dim):
                MAX, MIN = self.channel_statistics[c]
                if MAX> MIN:
                    img[:,:,c] = (img[:,:,c] + 0.5) * (MAX - MIN)   + MIN
                else:
                    img[:,:,c] =  MIN + 0.5
        elif len(img.shape) ==2:
            img = (img+ 0.5) * (self.MAX - self.MIN).numpy()
            img = img  + self.MIN
            return img 
        else:
            raise NotImplementedError()
        return img 

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


if __name__ == "__main__":
    compute_image_gradient(np.random.random(size=(10,10)))
    compute_image_divergence(np.random.random(size=(5,5))*10)