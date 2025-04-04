import yaml
import torch
from torch.utils.data import DataLoader
from models.network import BFANet
from data.dataset import BFADataset
from utils.trainer import train_model, value_check

def main(config):
    device = torch.device(f'cuda:{config["gpu_id"]}' if torch.cuda.is_available() and config["gpu_id"] is not None else 'cpu')

    train_dataset = BFADataset(config['train_csv_path'], config['json_dir'], task=config['task'])
    valid_dataset = BFADataset(config['valid_csv_path'], config['json_dir'], task=config['task'])
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True)

    model = BFANet(input_dim=128, task=config['task'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    train_model(model, train_loader, valid_loader, task=config['task'], 
                optimizer=optimizer, num_epochs=config['num_epochs'], device=device, 
                output_dir=config['output_dir'])


if __name__ == "__main__":
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    value_check(config)
    main(config)