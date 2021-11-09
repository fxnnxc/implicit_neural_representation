import numpy as np 
import scipy 
import torch 

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


if __name__ == "__main__":
    compute_image_gradient(np.random.random(size=(10,10)))
    compute_image_divergence(np.random.random(size=(5,5))*10)