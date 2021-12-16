# -----------------------------
# 2021.11.24 Bumjin Park
# -----------------------------

from typing import MutableMapping
from implicit_learning.trainer import PlotTrainer
from implicit_learning.model import  Siren, ReLU_PE_Model, ReLU_Model
from implicit_learning.dataset import *
from implicit_learning.utils import *
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from torch.utils.data import DataLoader 
from PIL import Image
import os 


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "2"

class MinMaxScaler():
    def __init__(self):
        self.MAX = 0 
        self.MIN = 0

    def fit_transform(self, img):
        self.MAX = 1
        self.MIN = 0
        assert self.MAX > self.MIN
        return (img - self.MIN)/(self.MAX-self.MIN)
    
    def inverse_transform(self, img):
        img = img * (self.MAX - self.MIN).numpy()
        img = img + img.min()
        return img 

class CompositePoissonEqn(Dataset):
    def __init__(self, config, transform, scaler1, scaler2):

        super().__init__()
        alpha = config.get('alpha')
        assert 0<=alpha <=1

        img1 = Image.open(config['data-path'])
        img1 = transform(img1)
        img1 = img1.mean(axis=0)
        img1 = scaler1.fit_transform(img1)
    
        img2 = Image.open(config['data-path2'])
        img2 = transform(img2)
        img2 = img2.mean(axis=0)
        img2 = scaler2.fit_transform(img2)
        
        img1 = img1.unsqueeze(0)
        img2 = img2.unsqueeze(0)
        
        sidelength_W = config.get("sidelength-W")
        sidelength_H = config.get("sidelength-H")
        # Compute gradient and laplacian       
        grads_x1, grads_y1 = compute_image_gradient(img1.numpy(), type=config.get("gradient_type"))
        grads_x1, grads_y1 = torch.from_numpy(grads_x1), torch.from_numpy(grads_y1)

        grads_x2, grads_y2 = compute_image_gradient(img2.numpy(), type=config.get("gradient_type"))
        grads_x2, grads_y2 = torch.from_numpy(grads_x2), torch.from_numpy(grads_y2)

        grads_x = alpha * grads_x1 + (1-alpha)* grads_x2
        grads_y = alpha * grads_y1 + (1-alpha)* grads_y2

        self.grads = torch.stack((grads_x, grads_y), dim=-1).view(-1,1, 2)
        self.laplace = scipy.ndimage.laplace(img1.numpy()).squeeze(0)[..., None]
        self.laplace = torch.from_numpy(self.laplace)
        
        self.pixels = alpha * img1.permute(1, 2, 0).view(-1, 1) +  (1-alpha)* img2.permute(1, 2, 0).view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.coords, {'pixels':self.pixels, 'grads':self.grads, 'laplace':self.laplace}


def construct_dataloader(config):
    sidelength = config['sidelength']
    scaler1 = MinMaxScaler()
    scaler2 = MinMaxScaler()
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        #Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])

    train = CompositePoissonEqn(config, transform=transform, scaler1=scaler1,scaler2=scaler2)
    valid = CompositePoissonEqn(config, transform=transform, scaler1=scaler1,scaler2=scaler2)
    test  = CompositePoissonEqn(config, transform=transform, scaler1=scaler1,scaler2=scaler2)
    
    train_dataloader =  DataLoader(train, batch_size=config.get("batch_size"), shuffle=True, pin_memory=False)
    valid_datalodaer =  DataLoader(valid, batch_size=config.get("batch_size"), shuffle=True, pin_memory=False)
    test_dataloader =   DataLoader(test, batch_size=config.get("batch_size"), shuffle=True, pin_memory=False)

    return train_dataloader, valid_datalodaer, test_dataloader, scaler1, scaler2 

class RGBTrainer(PlotTrainer):
    def __init__(self,  model, train_dataloader, valid_dataloader, test_dataloader, scaler1, scaler2, config, beta=0.8):
        super().__init__(model, train_dataloader, valid_dataloader, test_dataloader, scaler1,  config, beta)

import argparse
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--channel-dim", type=int)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--gradient-type", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--plot-epoch", type=int)
    parser.add_argument("--save-dir", type=str)
    parser.add_argument("--image-path", type=str)
    parser.add_argument("--image-path2", type=str)
    parser.add_argument("--image-length", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--lr-end", type=float)
    parser.add_argument("--hidden-layers", type=int)
    parser.add_argument("--hidden-features", type=int)
    parser.add_argument("--relu-pe-freq", type=int)
    parser.add_argument("--high-resolution", type=float, default=0)
    parser.add_argument("--plot-full", action="store_true")
    parser.add_argument("--plot-each", action="store_true")

args = parser.parse_args()

config = {
    "model":{
        "in_features":2,
        "hidden_features":args.hidden_features,
        "hidden_layers":args.hidden_layers,
        "out_features":args.channel_dim
    },
    "beta":args.beta,
    "gradient_type": args.gradient_type,
    "sidelength":args.image_length,
    "epochs":args.epochs,
    "lr":args.lr,
    "lr_end":args.lr_end,
    "batch_size":1,
    "alpha":args.alpha,
    "data-path":args.image_path,
    "data-path2":args.image_path2,
    "plot_epoch":args.plot_epoch,
    "save_dir":"checkpoint/"+f"{args.model}_epochs:{args.epochs}_beta:{args.beta}_alpha:{args.alpha}_layers:{args.hidden_layers}_dim:{args.hidden_features}_{args.save_dir}",
    "high_resolution": args.high_resolution,
    "plot_full":args.plot_full,
    "plot_each":args.plot_each
}
if args.model =="relu_pe":
    if args.relu_pe_freq is None:
        raise ValueError("use --relu-pe-freq ")
    config['model']['L'] = args.relu_pe_freq
    config['save_dir'] = config['save_dir'] + "_"+ str(args.relu_pe_freq)
    model = ReLU_PE_Model(**config['model'])
elif args.model =="relu":
    model = ReLU_Model(**config['model'])
elif args.model =="siren":
    model = Siren(**config['model'])
else:
    raise ValueError()

if not os.path.isdir("checkpoint"):
    os.mkdir("checkpoint")
if not os.path.isdir(config['save_dir']):
    os.mkdir(config['save_dir'])

model.cuda()
trainer = RGBTrainer(model, *construct_dataloader(config), config)
trainer.train()