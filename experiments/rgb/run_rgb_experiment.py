# -----------------------------
# 2021.11.24 Bumjin Park
# -----------------------------

from typing import MutableMapping
from implicit_learning.trainer import PlotTrainer
from implicit_learning.model import  Siren, ReLU_PE_Model, ReLU_Model
from implicit_learning.dataset import PoissonEqnRGB
from implicit_learning.utils import *
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from torch.utils.data import DataLoader 
import os 


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "2"


def construct_dataloader(config):
    sidelength = config['sidelength']
    transform = Compose([
        Resize(sidelength),
        ToTensor(),
        Normalize(torch.Tensor([0.5]), torch.Tensor([0.5]))
    ])

    train = PoissonEqnRGB(config, transform=transform)
    valid = PoissonEqnRGB(config, transform=transform)
    test  = PoissonEqnRGB(config, transform=transform)
    
    train_dataloader =  DataLoader(train, batch_size=config.get("batch_size"), shuffle=True, pin_memory=True)
    valid_datalodaer =  DataLoader(valid, batch_size=config.get("batch_size"), shuffle=True, pin_memory=True)
    test_dataloader =   DataLoader(test, batch_size=config.get("batch_size"), shuffle=True, pin_memory=True)

    return train_dataloader, valid_datalodaer, test_dataloader

class RGBTrainer(PlotTrainer):
    def __init__(self,  model, train_dataloader, valid_dataloader, test_dataloader, config, beta=0.8):
        super().__init__(model, train_dataloader, valid_dataloader, test_dataloader, config, beta)

import argparse
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--channel-dim", type=int)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--gradient-type", type=str)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--plot-epoch", type=int)
    parser.add_argument("--save-dir", type=str)
    parser.add_argument("--image-path", type=str)
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
    "data-path":args.image_path,
    "plot_epoch":args.plot_epoch,
    "save_dir":"checkpoint/"+f"{args.model}_epochs:{args.epochs}_beta:{args.beta}_layers:{args.hidden_layers}_dim:{args.hidden_features}_{args.save_dir}",
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