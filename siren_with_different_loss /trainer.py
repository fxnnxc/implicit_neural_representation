import torch 
import torch.nn  as nn 
class Trainer():
    def __init__(self, model, train_dataloader, valid_dataloader, test_dataloader, config=None):
        self.model = model
        self.epochs = 1000 
        self.lr = 1e-5 

        self.criterion = self.construct_criterion() 
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.trainloader = train_dataloader
        self.validloader = valid_dataloader
        self.testloader = test_dataloader



    def construct_criterion(self):
        return nn.L1Loss()

    def compute_loss(self, outputs, labels):
        return  self.criterion(outputs, labels)

    def train(self):
        print("train is started!")
        for epoch in range(self.epochs):
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


    