# -------------------------
# 2021.11.23 Bumjin Park 
# -------------------------

import torch 
import torch.nn  as nn 
from tqdm import tqdm 
from implicit_learning.utils import *
import matplotlib.pyplot as plt 

class Trainer():
    def __init__(self, model, train_dataloader, valid_dataloader, test_dataloader, config):
        self.model = model
        self.epochs = config.get("epochs") 
        self.lr = config.get("lr") 

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
    def __init__(self, model, train_dataloader, valid_dataloader, test_dataloader, config):
        super().__init__(model, train_dataloader, valid_dataloader, test_dataloader, config)
    
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