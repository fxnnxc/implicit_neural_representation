# -------------------------
# 2021.11.23 Bumjin Park 
# -------------------------

import numpy as np 
import scipy 
import torch 
import math
from skimage.metrics import structural_similarity as ssim


def compute_image_gradient(im_array):
    grads_x = scipy.ndimage.sobel(im_array, axis=0)[..., None]
    grads_y = scipy.ndimage.sobel(im_array, axis=1)[..., None]
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

if __name__ == "__main__":
    compute_image_gradient(np.random.random(size=(10,10)))
    compute_image_divergence(np.random.random(size=(5,5))*10)