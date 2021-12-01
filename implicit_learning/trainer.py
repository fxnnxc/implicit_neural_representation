# -------------------------
# 2021.11.23 Bumjin Park 
# -------------------------

import torch 
import torch.nn  as nn 
from tqdm import tqdm 
from implicit_learning.utils import *
import matplotlib.pyplot as plt 

import pandas as pd 
import torch
import matplotlib.pyplot as plt 
import copy
from implicit_learning.utils import *
from tqdm import tqdm 

## TODO---------------
##  combine the PlotTrainer -> Trainer

class Trainer():
    def __init__(self, model, train_dataloader, valid_dataloader, test_dataloader, scaler, config):
        self.model = model
        self.epochs = config.get("epochs") 
        self.lr = config.get("lr") 
        self.scaler = scaler

        self.criterion = self.construct_criterion() 
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.trainloader = train_dataloader
        self.validloader = valid_dataloader
        self.testloader = test_dataloader

        self.lambda_rgb = 0.3 
        self.lambda_grad = 0.3 
        self.lambda_lap = 0.4

        self.device  = torch.device("cuda" if config.get("gpu", False) else 'cpu')
        self.gpu = config.get("gpu", False)

    def construct_criterion(self):
        return nn.L1Loss()

    def compute_loss(self, outputs, labels):
        return  self.criterion(outputs, labels)

    def train(self):
        if self.gpu:
            self.model.cuda(self.device)
            print(torch.cuda.is_available())
        print("train is started!")
        for epoch in tqdm(range(self.epochs)):
            self.running_loss = 0.0 
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data 

                self.optimizer.zero_grad() 
                outputs =self.model(inputs)
                self.loss = self.compute_loss(outputs, labels) 
                self.loss.backward()
                self.optimizer.step()

                self.running_loss += self.loss.item()
            self.print_stats(epoch, i)
    
    def test(self):
        return 

    def print_stats(self, epoch, i):
        print("[%d %4d] loss : %.3f"%(epoch+1, i+1, self.running_loss/len(self.trainloader)))

class PoissonTrainer(Trainer):
    def __init__(self, model, train_dataloader, valid_dataloader, test_dataloader, scaler, config):
        super().__init__(model, train_dataloader, valid_dataloader, test_dataloader, scaler, config)
    
        model_input, gt = next(iter(train_dataloader))
        self.gt = {key: value.cuda() for key, value in gt.items()}
        self.model_input = model_input.cuda()

    def compute_loss(self, outputs, coords):
        train_loss = self.gradients_mse(outputs, coords, self.gt['grads'])
        return train_loss 

    def gradients_mse(self, model_output, coords, gt_gradients):
        # compute gradients on the model
        gradients = gradient(model_output, coords)
        gradients_loss = torch.mean((gradients - gt_gradients).pow(2).sum(-1))
        return gradients_loss

    def train(self):
        print("train is started!")
        for epoch in range(self.epochs):
            inputs = self.model_input

            self.optimizer.zero_grad() 
            outputs, coords =self.model(inputs)
            self.loss = self.compute_loss(outputs, coords) 
            self.loss.backward()
            self.optimizer.step()

            if epoch %100 == 0:
                print("Epoch %4d"%(epoch), f" : Loss : {self.loss}")
                img_grad = gradient(outputs, coords)
                self.plot(img_grad, self.gt)
    
    def plot(self, img_grad, gt):

        fig, axes = plt.subplots(1, 2, figsize=(4, 2))
        # axes[0].imshow(model_output.cpu().view(128,128).detach().numpy())
        # axes[1].imshow(original.view(128,128))
        axes[0].imshow(img_grad.cpu().norm(dim=-1).view(128,128).detach().numpy())
        axes[1].imshow(gt['grads'].cpu().norm(dim=-1).view(128,128).detach().numpy())
        plt.show()




class PlotTrainer(PoissonTrainer):
    def __init__(self, model, train_dataloader, valid_dataloader, test_dataloader, scaler, config, beta=0.8):
        self.lr = config.get("lr")
        self.lr_end = config.get("lr_end")
        self.save_dir = config.get("save_dir")
        super().__init__(model, train_dataloader, valid_dataloader, test_dataloader, scaler, config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.plot_epoch = config.get("plot_epoch")
        self.sidelength = config.get("sidelength")
        self.channel_dim = config.get("model")['out_features']
        self.size = (self.sidelength, self.sidelength, self.channel_dim)

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer,
                                        lr_lambda=lambda epoch: (self.lr + (self.lr_end - self.lr)/self.epochs * epoch),
                                        last_epoch=-1,
                                        verbose=False)
        self.beta = config.get("beta") 
        self.df = pd.DataFrame(columns=["grad_loss", "value_loss", "psnr", "ssim", "lr"])        
        self.high_resolution = config.get("high_resolution")
        self.plot_full = config.get("plot_full", False)
        self.plot_each = config.get("plot_each", False)
        
        if self.plot_full:  
            self.model_outputs = [] 
            self.model_gradients = [] 
            self.high_resolutions = []
            self.plot_epochs = []  


    def train(self):
        print("===== train is started! =====")
        for epoch in tqdm(range(self.epochs+1)):
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


# ------------------------------------
# computation of the loss 
    def compute_value_loss(self, outputs, coords):
        train_loss = ((outputs - self.gt['pixels'])**2).mean()
        return train_loss

    def compute_grad_loss(self, outputs, coords):
        v = self.gt['grads']
        train_loss = self.gradients_mse(outputs, coords, v)
        return train_loss 

    def gradients_mse(self, model_output, coords, gt_gradients):
        gradients = self.rgb_gradient(model_output, coords)
        gradients_loss = 0
        self.img_grad = [] 
        for i, grad in enumerate(gradients):
            gradients_loss += torch.mean((grad - gt_gradients[:,:,i,:]).pow(2).sum(-1))
            self.img_grad.append(grad.clone().detach().cpu())
        return gradients_loss

    def rgb_gradient(self, y, x, grad_outputs=None):
        grads =[]
        y = y.view(1, -1, self.channel_dim)
        grad_outputs = [torch.ones_like(y[0,:,i]) for i in range(self.channel_dim)]
        for i in range(self.channel_dim):
            grads.append(torch.autograd.grad(y[0,:,i], [x], grad_outputs=grad_outputs[i], create_graph=True)[0])
        return grads

# ---------------------------------------------------
# Additional Methods including psnr, ssim, and plotting

    def detach_and_calculate_psnr(self, model_output, ground_truth):
        model_output = model_output.view(*self.size).clone().detach().cpu().numpy()
        ground_truth = ground_truth.view(*self.size).clone().detach().cpu().numpy()
        return calculate_psnr(model_output, ground_truth)

    def detach_and_calculate_ssim(self, model_output, ground_truth):
        model_output = model_output.view(*self.size).clone().detach().cpu().numpy()
        ground_truth = ground_truth.view(*self.size).clone().detach().cpu().numpy()
        return calculate_ssim(model_output, ground_truth) 

    def generate_higher_resolution(self, scale=2, dim=2):
        # given the model and the original codors, generate scale times resolution
        tensors = tuple(dim * [torch.linspace(-1, 1, steps=int(self.sidelength*scale))])
        mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
        mgrid = mgrid.reshape(1,-1, dim).cuda()
        with torch.no_grad():
            hs_image, coords = self.model(mgrid)
        return hs_image, int(self.sidelength*scale)

    def save_for_full_plot(self, model_output, img_grad, epoch, high_resolution=None, hs_sidelength=None):
        self.model_outputs.append(model_output.cpu().view(*self.size).numpy())
        self.model_gradients.append([grad.cpu().view(*(self.size[:2]),2)/self.sidelength for grad in img_grad])
        self.plot_epochs.append(epoch)
        if self.high_resolution:
            self.high_resolutions.append(high_resolution.norm(dim=-1).view(hs_sidelength, hs_sidelength, self.channel_dim).numpy())
            self.hs_sidelength= hs_sidelength

    def store_full_plot(self):
        form = "png"
        # --- image plot
        LENGTH = len(self.plot_epochs)
        if len(self.high_resolutions)==0:
            fig, axes = plt.subplots(LENGTH, 2, figsize=(LENGTH*2, 4*(LENGTH)))
        else:
            fig, axes = plt.subplots(LENGTH, 3, figsize=(LENGTH*2, 4*(LENGTH)))
            axes[0, 2].set_title("HS X %.1f"%(self.hs_sidelength/self.sidelength))
            for i in range(LENGTH):
                axes[i, 2].imshow(self.high_resolutions[i])
                
        axes[0,0].set_title("Pred values", fontsize=20)
        axes[0,1].set_title("GT values", fontsize=20)
        for i in range(LENGTH):
            model_image = copy.deepcopy(self.model_outputs[i].reshape(*self.size))
            for c in range(self.channel_dim):
                model_image[:,:,c] = (model_image[:,:,c] - model_image[:,:,c].min())/(model_image[:,:,c].max() - model_image[:,:,c].min())
            axes[i,0].imshow(model_image)
            axes[i,1].imshow(self.gt['pixels'].clone().detach().cpu().view(*self.size).numpy(), vmin= model_image.min(), vmax=model_image.max())
            axes[i,0].set_ylabel(self.plot_epochs[i], rotation=0,fontsize=20)
        plt.tight_layout()
        plt.show()
        plt.savefig(self.save_dir +f"/image_full.{form}")
        plt.close(fig)
        
        # --- gradient plot 
        fig, axes = plt.subplots(LENGTH, 2*self.channel_dim, figsize=(LENGTH*2*self.channel_dim, 4*(LENGTH)))
        for i in range(LENGTH):
            axes[i,0].set_ylabel(self.plot_epochs[i], rotation=0,fontsize=20)   
            for k in range(self.channel_dim):
                axes[0,2*k].set_title(f"Ch:{k} Pred grads", fontsize=20)
                axes[0,2*k+1].set_title(f"Ch:{k} GT grads", fontsize=20)   
                axes[i,2*k].imshow(self.model_gradients[i][k].norm(dim=-1))
                axes[i,2*k+1].imshow(self.gt['grads'][:,:,k,:].clone().detach().cpu().norm(dim=-1).view(*(self.size[:2])).numpy()/self.sidelength)
                
        plt.tight_layout()
        plt.show()
        plt.savefig(self.save_dir +f"/grads_full.{form}")
        plt.close(fig)

        # --- line plot
        fig, axes = plt.subplots(LENGTH, 2, figsize=(LENGTH*2, 4*(LENGTH)))
        dim, x_line, y_line = 0, 50, 50
        for i in range(LENGTH):
            # --- x
            pixel_model = self.model_outputs[i][x_line,:,dim]
            pixel_truth = self.gt['pixels'].clone().detach().cpu().view(*self.size).numpy()[x_line, :, dim]
            cum_sum_truth = np.cumsum(self.gt['grads'].clone().detach().cpu().view(*self.size, 2).numpy()[x_line,:,dim,1]/self.sidelength)
            cum_sum_model =np.cumsum(self.model_gradients[i][dim].clone().view(*(self.size[:2]),2).numpy()[x_line,:,1])
            # ---- offset 0
            pixel_model -= pixel_model[0]
            pixel_truth -= pixel_truth[0]
            cum_sum_truth -= cum_sum_truth[0]
            cum_sum_model -= cum_sum_model[0]
            axes[i,0].plot(copy.deepcopy(pixel_model))
            axes[i,0].plot(copy.deepcopy(pixel_truth))
            axes[i,0].plot(copy.deepcopy(cum_sum_truth))
            axes[i,0].plot(copy.deepcopy(cum_sum_model))
            axes[i,0].set_ylabel(self.plot_epochs[i], rotation=0,fontsize=20)
        
            # --- y
            pixel_model = self.model_outputs[i][:, y_line, dim]
            pixel_truth = self.gt['pixels'].clone().detach().cpu().view(*self.size).numpy()[:, y_line, dim]
            cum_sum_truth = np.cumsum(self.gt['grads'].clone().detach().cpu().view(*self.size,2).numpy()[:,y_line,dim,0]/self.sidelength)
            cum_sum_model = np.cumsum(self.model_gradients[i][dim].clone().view(*(self.size[:2]),2).numpy()[:,y_line,0])
            # ---- offset 0
            pixel_model -= pixel_model[0]
            pixel_truth -= pixel_truth[0]
            cum_sum_truth -= cum_sum_truth[0]
            cum_sum_model -= cum_sum_model[0]
            axes[i,1].plot(copy.deepcopy(pixel_model))
            axes[i,1].plot(copy.deepcopy(pixel_truth))
            axes[i,1].plot(copy.deepcopy(cum_sum_truth))
            axes[i,1].plot(copy.deepcopy(cum_sum_model))
            axes[i,1].set_ylabel(self.plot_epochs[i], rotation=0,fontsize=20)
    
        axes[0,0].legend(["model_out", "GT", "cum_sum(gt)", "cum_sum(model)"])
        axes[0,0].set_title(f"X_line: {x_line}", fontsize=20)
        axes[0,1].legend(["model_out", "GT", "cum_sum(gt)", "cum_sum(model)"])
        axes[0,1].set_title(f"Y_line: {y_line}", fontsize=20)
        plt.tight_layout()
        plt.show()
        plt.savefig(self.save_dir +f"/line_full.{form}")
        plt.close(fig)

        # --- histogram 
        fig, axes = plt.subplots(LENGTH, 2*self.channel_dim, figsize=(LENGTH*2, 4*(LENGTH)))
        for i in range(LENGTH):
            axes[i,0].set_ylabel(self.plot_epochs[i], rotation=0,fontsize=20)   
            for k in range(self.channel_dim):
                axes[i,2*k].hist(self.model_outputs[i].flatten(), bins=200,log=True, alpha=0.5)
                axes[i,2*k].hist(self.gt['pixels'][:,:,k].clone().detach().cpu().flatten().cpu().detach().numpy(), bins=200,log=True, alpha=0.5)
                axes[i,2*k+1].hist(self.model_gradients[i][k].flatten().cpu().numpy(), bins=200,log=True, alpha=0.5)
                axes[i,2*k+1].hist(self.gt['grads'][:,:,k,:].clone().detach().cpu().flatten().numpy()/self.sidelength, bins=200,log=True, alpha=0.5)

                axes[0,0].legend(["model", "ground truth"])
                axes[0,2*k].set_title(f"CH:{k} | value histogram", fontsize=20)
                axes[0,2*k+1].set_title(f"CH:{k} | grad histogram", fontsize=20)
        plt.tight_layout()
        plt.show()
        plt.savefig(self.save_dir +f"/histogram_full.{form}")
        plt.close(fig)

    def plot(self, model_output, original, img_grad, gt, epoch, high_resolution=None, hs_sidelength=None):
        ## 
        import warnings
        warnings.warn("It will be deprecated in the furue", UserWarning)
        form = "png"
        # --- image plot
        if high_resolution is None:
            fig, axes = plt.subplots(1, 4, figsize=(10, 5))
        else:
            fig, axes = plt.subplots(1, 5, figsize=(10, 5))
            axes[4].imshow(high_resolution.norm(dim=-1).view(hs_sidelength, hs_sidelength).numpy())
            axes[4].set_title("HS X %.1f"%(hs_sidelength/self.sidelength))

        axes[0].imshow(model_output.view(*self.size).numpy())
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
        axes[0].plot(np.cumsum(gt['grads'].clone().detach().cpu().view(self.sidelength,self.sidelength,2).numpy()[x_line,:,1] ))
        axes[0].legend(["model_out", "gt", "cum_sum"])
        axes[0].set_title(f"X_line: {x_line}")

        axes[1].plot(model_output.view(self.sidelength,self.sidelength).numpy()[:,y_line])
        axes[1].plot(original.view(self.sidelength,self.sidelength).numpy()[:,y_line])
        axes[1].plot(np.cumsum(gt['grads'].clone().detach().cpu().view(self.sidelength,self.sidelength,2).numpy()[:,y_line,0]))
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
        axes[1].hist(img_grad.flatten().numpy(), bins=200,log=True, alpha=0.5)
        axes[1].hist(gt['grads'].clone().detach().cpu().view(1, self.sidelength, self.sidelength,2)[:,:,:,0].flatten().numpy(), bins=200,log=True, alpha=0.5)
        axes[1].legend(["model", "ground truth"])
        axes[1].set_title("grad histogram")
        plt.tight_layout()
        plt.savefig(self.save_dir +f"/histogram_{epoch}.{form}")
        plt.close(fig)