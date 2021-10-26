from dataset import CustomDataset
from trainer import Trainer 
from model import Model
from torch.utils.data import DataLoader 
from torchvision import transforms

import argparse 

def construct_dataloader():
    train_dataset = CustomDataset(transform=transforms.ToTensor())
    valid_dataset = CustomDataset(transform=transforms.ToTensor())
    test_dataset = CustomDataset(transform=transforms.ToTensor())

    train_dataloader =  DataLoader(train_dataset, batch_size=32, shuffle=True, )
    valid_datalodaer =  DataLoader(valid_dataset, batch_size=32, shuffle=True)
    test_dataloader =   DataLoader(test_dataset, batch_size=32, shuffle=True)

    return train_dataloader, valid_datalodaer, test_dataloader


def main(config):
    model = Model(config['model'])
    trainer = Trainer(model, *construct_dataloader(), config)
    trainer.train()

if __name__ == "__main__":
    import argparse, json  
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)

    main(config)