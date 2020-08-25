import os
import wandb
import torch
import torch.nn.functional as F
from torch import nn, optim
import pandas as pd
import numpy as np
import argparse
import matplotlib
import random
matplotlib.use('agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from dataloader import AnomalyDataset, get_loader
from model import *

from pathlib import Path

RANDOM_SEED = 42

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def evaluate(model, val_dl, optimizer, criterion):
    val_losses = []
    if model.training:
        model.eval()
    with torch.no_grad():      
        for step, mb in enumerate(tqdm(val_dl)):
            inputs = mb.cuda()
            outputs = model(inputs)
            val_loss = criterion(outputs, inputs.squeeze(1))
            val_losses.append(val_loss.item())
                
    val_loss = np.mean(val_losses)

    return val_loss


def train(model, tr_dl, optimizer, criterion):
    if not model.training:
        model.train()
    best_loss = np.inf
    tr_mse, val_mse = [], []
    for epoch in tqdm(range(args.num_epochs)):
        tr_losses = []
        tr_step = 0
        for step, mb in enumerate(tqdm(tr_dl)):
            inputs = mb.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)

            mb_loss = criterion(outputs, inputs.squeeze(1))
            mb_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            
            tr_losses.append(mb_loss.item())
            tr_step += mb_loss.item()
            if step % 100 == 0:
                wandb.log({"epoch": epoch + 1, "loss": tr_step / (step + 1)})
                print(f"Epoch: {epoch + 1}, tr_loss: {tr_step / (step + 1):.5f}")
        
        else:
            tr_loss = np.mean(tr_losses)
            val_loss = evaluate(model, val_dl, optimizer, criterion)
            
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(model.state_dict(), f'weights/ec2_transformer.pt')
                print("Save the weight!")
          
            wandb.log({"epoch": epoch + 1, "tr_loss": tr_loss, "val_loss": val_loss})
            print(f"Epoch: {epoch + 1}, tr_loss: {tr_loss}, val_loss: {val_loss}")
        scheduler.step()

        model.train()
        tr_mse.append(tr_loss)
        val_mse.append(val_loss)
    return tr_mse, val_mse


def main(args):
    tr_mse, val_mse = train(model, tr_dl, optimizer, criterion) 
    return tr_mse, val_mse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # dataloader config
    parser.add_argument("--batch_size", default=48, type=int, help="Size of mini batch")
    
    # metric config

    parser.add_argument("--num_epochs", default=20, type=int, help="Number of total training epochs")
    parser.add_argument("--learning_rate", default=1e-3, type=float, help="learning rate")
    
    # model config
    parser.add_argument("--num_features", default=1, type=int, help="Number of features")
    parser.add_argument("--num_layers", default=1, type=int, help="Number of layers")
    parser.add_argument("--num_classes", default=1, type=int, help="Number of classes")
    parser.add_argument("--hidden_size", default=10, type=int, help="Number of hidden_size")
    parser.add_argument("--device", default=5, type=int, help="device number of gpu")

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    # model = StackedGRU(1, 1, 1)
    # model = LSTM(1, 1, 1, 1, 1, bidirectional=True)
    # model = LstmAutoEncoder(1, 1, 32, 1)
    # model = SAE(1)
    model = TransAm(feature_size=10)

    model.cuda()

    criterion = nn.MSELoss()
    # criterion = nn.L1Loss(reduction='sum')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.98)

    wandb.init(sync_tensorboard=True)
    wandb.watch(model)

    # tr_path = "/daintlab/data/anomaly_detection/NAB-master/data/realAWSCloudwatch/ec2_network_in_257a54.csv"
    tr_path = "/daintlab/data/anomaly_detection/NAB-master/data/realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv"
    # tr_filename = "realAWSCloudwatch/ec2_network_in_257a54.csv"
    tr_filename = "realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv"
    # ec2_network_in_257a54

    tr_ds = AnomalyDataset(tr_path, tr_path, tr_filename, tr_filename)
    tr_dl = DataLoader(tr_ds[:3225].reshape(-1, 1, 1), shuffle=False, batch_size=args.batch_size) 

    # val_ds = AnomalyDataset(tr_path, tr_path, tr_filename, tr_filename)
    val_dl = DataLoader(tr_ds[3225:].reshape(-1, 1, 1), shuffle=False, batch_size=args.batch_size)

    # tr_ds, tr_dl, tr_df = get_loader(tr_path, tr_path, tr_filename, tst_filename, batch_size=args.batch_size)

    # tst_ds = AnomalyDataset(tr_path, tst_path)
    # tst_dl = DataLoader(tst_ds, shuffle=False, batch_size=args.batch_size)


    tr_mse, val_mse = main(args)
    print(tr_mse)
    print(val_mse)

 