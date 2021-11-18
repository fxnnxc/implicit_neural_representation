from implicit_learning.dataset import CustomDataset, SingleImageDataset
from implicit_learning.trainer import Trainer 
from implicit_learning.model import TestModel
from torch.utils.data import DataLoader 
from torchvision import transforms
import argparse 
import torch
import os 


def construct_dataloader(config):
    data_type =config['data-type']
    print(data_type)
    
    if data_type == "test":
        train = CustomDataset(transform=transforms.ToTensor())
        valid = CustomDataset(transform=transforms.ToTensor())
        test = CustomDataset(transform=transforms.ToTensor())

    elif data_type =="single-image":
        train = SingleImageDataset(config, type="train", transform=transforms.ToTensor())
        valid = SingleImageDataset(config, type="valid",  transform=transforms.ToTensor())
        test = SingleImageDataset(config, type="test", transform=transforms.ToTensor())
    else:
        ValueError("please use --datatype XXX")

    train_dataloader =  DataLoader(train, batch_size=config.get("batch_size"), shuffle=True)
    valid_datalodaer =  DataLoader(valid, batch_size=config.get("batch_size"), shuffle=True)
    test_dataloader =   DataLoader(test, batch_size=config.get("batch_size"), shuffle=True)

    return train_dataloader, valid_datalodaer, test_dataloader


def main(config):
    model = TestModel(config['model'])
    trainer = Trainer(model, *construct_dataloader(config), config)
    trainer.train()

if __name__ == "__main__":
    import argparse, json  
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='config.json')
    parser.add_argument("--data-type", type=str)
    parser.add_argument("--data-path", type=str)
    parser.add_argument("--gpu", action="store_true")

    args = parser.parse_args()
    with open(args.config) as f:
        config = json.load(f)
    config['data-path'] = args.data_path
    config['data-type'] =args.data_type
    main(config)

