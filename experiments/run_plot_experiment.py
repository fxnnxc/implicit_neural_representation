# -----------------------------
# 2021.11.24 Bumjin Park
# -----------------------------

from typing import MutableMapping
from implicit_learning.trainer import PoissonTrainer 
from implicit_learning.model import  Siren, ReLU_PE_Model, ReLU_Model
from implicit_learning.dataset import PoissonEqn
from implicit_learning.utils import *
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
from torch.utils.data import DataLoader 
import pandas as pd 
import torch
import matplotlib.pyplot as plt 
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

    train = PoissonEqn(config, transform=transform)
    valid = PoissonEqn(config, transform=transform)
    test  = PoissonEqn(config, transform=transform)
    
    train_dataloader =  DataLoader(train, batch_size=config.get("batch_size"), shuffle=True, pin_memory=True)
    valid_datalodaer =  DataLoader(valid, batch_size=config.get("batch_size"), shuffle=True, pin_memory=True)
    test_dataloader =   DataLoader(test, batch_size=config.get("batch_size"), shuffle=True, pin_memory=True)

    return train_dataloader, valid_datalodaer, test_dataloader

class CustomizeTrainer(PoissonTrainer):
    def __init__(self, model, train_dataloader, valid_dataloader, test_dataloader, config, beta=0.8):
        self.lr = config.get("lr")
        self.lr_end = config.get("lr_end")
        self.save_dir = config.get("save_dir")
        super().__init__(model, train_dataloader, valid_dataloader, test_dataloader, config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.plot_epoch = config.get("plot_epoch")
        self.sidelength = config.get("sidelength")

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                        lr_lambda=lambda epoch: (self.lr + (self.lr_end - self.lr)/self.epochs * epoch),
                                        last_epoch=-1,
                                        verbose=False)
        self.beta = config.get("beta") 
        self.df = pd.DataFrame(columns=["grad_loss", "value_loss", "psnr", "ssim", "lr"])        
        self.high_resolution = config.get("high_resolution")
        self.plot_full = config.get("plot_full", False)
        self.plot_each = config.get("plot_each", False)
        if config.get("gradient_type") =="sobel":
            self.multiplier = 2*4*4
        elif config.get("gradient_type")=="convolve":
            self.multiplier = 2  
        elif config.get("gradient_type")=="bumjin":
            self.multiplier = (2*np.sqrt(2)) * (2*np.sqrt(2)+2) * (2*np.sqrt(2)+2) 
        else:
            raise ValueError("unsupported gradient type")

        if self.plot_full:  
            self.model_outputs = [] 
            self.model_gradients = [] 
            self.high_resolutions = []
            self.plot_epochs = []  



    def train(self):
        print("===== train is started! =====")
        for epoch in range(self.epochs+1):
            inputs = self.model_input
            self.optimizer.zero_grad() 
            outputs, coords =self.model(inputs)
            grad_loss = self.compute_grad_loss(outputs, coords)
            value_loss = self.compute_value_loss(outputs, coords)
            self.loss = (1- self.beta) * grad_loss + self.beta * value_loss

            if epoch %  self.plot_epoch == 0:
                outputs_for_others = outputs.clone().detach().cpu()
                psnr = self.detach_and_calculate_psnr(outputs_for_others, self.gt['pixels'])
                ssim = self.detach_and_calculate_ssim(outputs_for_others, self.gt['pixels'])
                print("--- Epoch %5d/%d  | lr: %.5f ---"%(epoch, self.epochs, (self.lr + (self.lr_end - self.lr)/self.epochs * epoch)))
                print(f"Beta: {self.beta:.5f}", end=" | ")
                print("Loss: %.6f  value_loss: %.5f  grad_loss: %.5f"%(self.loss, value_loss.item(), grad_loss.item()))
                print("PSNR : %.3f | SSIM : %.3f |"%(psnr, ssim))
                print("------------------------------------------")                                                    
                
                hs_image, hs_sidelength = None, None 
                if self.high_resolution>0:
                    hs_image, hs_sidelength = self.generate_higher_resolution(scale=self.high_resolution, dim=2)
                self.df.loc[epoch, :] = {
                    "grad_loss":  grad_loss.item(),
                    "value_loss" : value_loss.item(),
                    "psnr": psnr,
                    "ssim": ssim,
                    "lr": (self.lr + (self.lr_end - self.lr)/self.epochs * epoch)
                }
                self.df.to_csv(self.save_dir+"/result.csv")
                if self.plot_each:
                    self.plot(outputs_for_others, self.gt['pixels'], self.img_grad, self.gt, epoch, hs_image, hs_sidelength)
                if self.plot_full:
                    self.save_for_full_plot(outputs_for_others, self.img_grad, epoch, hs_image, hs_sidelength)

            self.loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        if self.plot_full:
            self.store_full_plot()

    def compute_grad_loss(self, outputs, coords):
        v = self.gt['grads']
        train_loss = self.gradients_mse(outputs, coords, v)
        return train_loss 

    def compute_value_loss(self, outputs, coords):
        train_loss = ((outputs - self.gt['pixels'])**2).mean()
        return train_loss

    def set_beta(self, beta):
        self.beta = beta 

    def gradients_mse(self, model_output, coords, gt_gradients):
        gradients = gradient(model_output, coords)
        gradients_loss = torch.mean((gradients - gt_gradients).pow(2).sum(-1))
        self.img_grad = gradients.clone().detach().cpu()

        return gradients_loss

    def detach_and_calculate_psnr(self, model_output, ground_truth):
        model_output = model_output.view(self.sidelength, self.sidelength).clone().detach().cpu().numpy()
        ground_truth = ground_truth.view(self.sidelength, self.sidelength).clone().detach().cpu().numpy()
        return calculate_psnr(model_output, ground_truth)

    def detach_and_calculate_ssim(self, model_output, ground_truth):
        model_output = model_output.view(self.sidelength, self.sidelength).clone().detach().cpu().numpy()
        ground_truth = ground_truth.view(self.sidelength, self.sidelength).clone().detach().cpu().numpy()
        return calculate_ssim(model_output, ground_truth) 

    def generate_higher_resolution(self, scale=2, dim=2):
        # given the model and the original codors, generate scale times resolution
        tensors = tuple(dim * [torch.linspace(-1, 1, steps=int(self.sidelength*scale))])
        mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
        mgrid = mgrid.reshape(1,-1, dim).cuda()
        with torch.no_grad():
            hs_image, coords = self.model(mgrid)
        return hs_image, int(self.sidelength*scale)

    def save_for_full_plot(self, model_output, img_grad, epoch, high_resolution=None, hs_sidelength=None ):
        self.model_outputs.append(model_output.cpu().view(self.sidelength,self.sidelength).numpy())
        self.model_gradients.append(img_grad.norm(dim=-1).view(self.sidelength,self.sidelength).numpy())
        self.plot_epochs.append(epoch)
        if self.high_resolution:
            self.high_resolutions.append(high_resolution.norm(dim=-1).view(hs_sidelength, hs_sidelength).numpy())
            self.hs_sidelength= hs_sidelength


    def store_full_plot(self):
        form = "png"
        # --- image plot
        LENGTH = len(self.plot_epochs)
        if len(self.high_resolutions)==0:
            fig, axes = plt.subplots(LENGTH, 4, figsize=(LENGTH*2, 4*(LENGTH)))
        else:
            fig, axes = plt.subplots(LENGTH, 5, figsize=(LENGTH*2, 4*(LENGTH)))
            axes[0, 4].set_title("HS X %.1f"%(self.hs_sidelength/self.sidelength))
            for i in range(LENGTH):
                axes[i, 4].imshow(self.high_resolutions[i])
                
        axes[0,0].set_title("Pred values", fontsize=20)
        axes[0,1].set_title("GT values", fontsize=20)
        axes[0,2].set_title("Pred grads", fontsize=20)
        axes[0,3].set_title("GT grads", fontsize=20)
        for i in range(LENGTH):
            axes[i,0].imshow(self.model_outputs[i])
            axes[i,1].imshow(self.gt['pixels'].clone().detach().cpu().view(self.sidelength,self.sidelength).numpy())
            axes[i,2].imshow(self.model_gradients[i])
            axes[i,3].imshow(self.gt['grads'].clone().detach().cpu().norm(dim=-1).view(self.sidelength,self.sidelength).numpy())
            axes[i,0].set_ylabel(self.plot_epochs[i], rotation=0,fontsize=20)

        plt.tight_layout()
        plt.savefig(self.save_dir +f"/image_full.{form}")
        plt.close(fig)
        
        # --- line plot
        fig, axes = plt.subplots(LENGTH, 2, figsize=(LENGTH*2, 4*(LENGTH)))
        x_line, y_line = 50, 50

        for i in range(LENGTH):
            axes[i,0].plot(self.model_outputs[i][x_line,:])
            axes[i,0].plot(self.gt['pixels'].clone().detach().cpu().view(self.sidelength,self.sidelength).numpy()[x_line,:])
            axes[i,0].plot(np.cumsum(self.gt['grads'].clone().detach().cpu().view(self.sidelength,self.sidelength,2).numpy()[x_line,:,1]/self.multiplier))#/self.sidelength ))
            axes[i,0].set_ylabel(self.plot_epochs[i], rotation=0,fontsize=20)
        
            axes[i,1].plot(self.model_outputs[i][:,y_line])
            axes[i,1].plot(self.gt['pixels'].clone().detach().cpu().view(self.sidelength,self.sidelength).numpy()[:,y_line])
            axes[i,1].plot(np.cumsum(self.gt['grads'].clone().detach().cpu().view(self.sidelength,self.sidelength,2).numpy()[:,y_line,0]/self.multiplier))#/self.sidelength ))
        axes[0,0].legend(["model_out", "GT", "cum_sum"])
        axes[0,0].set_title(f"X_line: {x_line}", fontsize=20)
        axes[0,1].legend(["model_out", "GT", "cum_sum"])
        axes[0,1].set_title(f"Y_line: {y_line}", fontsize=20)
        plt.tight_layout()

        plt.savefig(self.save_dir +f"/line_full.{form}")
        plt.close(fig)

        # --- histogram 
        fig, axes = plt.subplots(LENGTH, 2, figsize=(LENGTH*2, 4*(LENGTH)))

        for i in range(LENGTH):
            axes[i,0].hist(self.model_outputs[i].flatten(), bins=200,log=True, alpha=0.5)
            axes[i,0].hist(self.gt['pixels'].clone().detach().cpu().flatten().cpu().detach().numpy(), bins=200,log=True, alpha=0.5)
            axes[i,1].hist(self.model_gradients[i].flatten(), bins=200,log=True, alpha=0.5)
            axes[i,1].hist(self.gt['grads'].clone().detach().cpu().view(1, self.sidelength, self.sidelength,2)[:,:,:,0].flatten().cpu().detach().numpy()/self.multiplier, bins=200,log=True, alpha=0.5)
            axes[i,0].set_ylabel(self.plot_epochs[i], rotation=0,fontsize=20)

        axes[0,0].legend(["model", "ground truth"])
        axes[0,0].set_title("value histogram", fontsize=20)
        axes[0,1].legend(["model", "ground truth"])
        axes[0,1].set_title("grad histogram", fontsize=20)
        plt.tight_layout()

        plt.savefig(self.save_dir +f"/histogram_full.{form}")
        plt.close(fig)

    def plot(self, model_output, original, img_grad, gt, epoch, high_resolution=None, hs_sidelength=None):
        form = "png"
        # --- image plot
        if high_resolution is None:
            fig, axes = plt.subplots(1, 4, figsize=(10, 5))
        else:
            fig, axes = plt.subplots(1, 5, figsize=(10, 5))
            axes[4].imshow(high_resolution.norm(dim=-1).view(hs_sidelength, hs_sidelength).numpy())
            axes[4].set_title("HS X %.1f"%(hs_sidelength/self.sidelength))

        axes[0].imshow(model_output.view(self.sidelength,self.sidelength).numpy())
        axes[1].imshow(original.view(self.sidelength,self.sidelength).numpy())
        axes[2].imshow(img_grad.norm(dim=-1).view(self.sidelength,self.sidelength).numpy())
        axes[3].imshow(gt['grads'].clone().detach().cpu().norm(dim=-1).view(self.sidelength,self.sidelength).numpy())
        axes[0].set_title("Pred values")
        axes[1].set_title("GT values")
        axes[2].set_title("Pred grads")
        axes[3].set_title("GT grads")
        plt.tight_layout()
        plt.savefig(self.save_dir +f"/image_{epoch}.{form}")
        plt.close(fig)

        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        x_line, y_line = 50, 50
        axes[0].plot(model_output.view(self.sidelength,self.sidelength).numpy()[x_line,:])
        axes[0].plot(original.view(self.sidelength,self.sidelength).numpy()[x_line,:])
        axes[0].plot(np.cumsum(gt['grads'].clone().detach().cpu().view(self.sidelength,self.sidelength,2).numpy()[x_line,:,1] /self.multiplier ))
        axes[0].legend(["model_out", "gt", "cum_sum"])
        axes[0].set_title(f"X_line: {x_line}")

        axes[1].plot(model_output.view(self.sidelength,self.sidelength).numpy()[:,y_line])
        axes[1].plot(original.view(self.sidelength,self.sidelength).numpy()[:,y_line])
        axes[1].plot(np.cumsum(gt['grads'].clone().detach().cpu().view(self.sidelength,self.sidelength,2).numpy()[:,y_line,0]/self.multiplier))
        axes[1].legend(["model_out", "gt", "cum_sum"])
        axes[1].set_title(f"Y_line: {y_line}")

        plt.savefig(self.save_dir +f"/line_{epoch}.{form}")
        plt.close(fig)

        # --- histogram 
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        axes[0].hist(model_output.flatten().numpy(), bins=200,log=True, alpha=0.5)
        axes[0].hist(gt['pixels'].clone().detach().cpu().flatten().numpy(), bins=200,log=True, alpha=0.5)
        axes[0].legend(["model", "ground truth"])
        axes[0].set_title("value histogram")
        axes[1].hist(img_grad.view(1, self.sidelength, self.sidelength,2)[:,:,:,0].flatten().numpy(), bins=200,log=True, alpha=0.5)
        axes[1].hist(gt['grads'].clone().detach().cpu().view(1, self.sidelength, self.sidelength,2)[:,:,:,0].flatten().numpy(), bins=200,log=True, alpha=0.5)
        axes[1].legend(["model", "ground truth"])
        axes[1].set_title("grad histogram")
        plt.tight_layout()
        plt.savefig(self.save_dir +f"/histogram_{epoch}.{form}")
        plt.close(fig)

import argparse
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
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
        "out_features":1
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
trainer = CustomizeTrainer(model, *construct_dataloader(config), config)
trainer.train()