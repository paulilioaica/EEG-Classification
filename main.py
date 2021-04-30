import json

from models.GCN import GCN
import torch
from dataloader import DEAP
from trainer import Trainer
from torch_geometric.data import DataLoader
import torch.optim as optim


def run(config, train_dataset, val_dataset):
    device = 'cpu' if torch.cuda.is_available() else 'cpu'
    model = GCN(1, 64, 4, 0.01).to(device)
    print("Training on {}, batch_size is {}, lr is {}".format(device, config['batch_size'], config['lr']))
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=True)

    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, config, device)
    train_acc, train_loss, val_acc, val_loss = trainer.train()
    return train_acc, train_loss, val_acc, val_loss


if __name__ == '__main__':
    with open('config.json', 'r') as f:
        config = json.load(f)
    train_dataset = DEAP(root_dir="./clean_data", label_path='clean_data')
    train_acc, train_loss, val_acc, val_loss = run(config, train_dataset, train_dataset)