import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from options.neural.simulator import SimulatorEuropean


class European_Sup_NN(nn.Module):
    def __init__(self):
        super(European_Sup_NN, self).__init__()
        self.seq = nn.Sequential(
            nn.Linear(6, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 512),
            nn.Tanh(),
            nn.Linear(512, 1)
        )

    def forward(self, inputs):
        outputs = self.seq(inputs).squeeze(1)
        return outputs

    
class CustomDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input = torch.tensor(self.df.iloc[idx][['S', 'K', 'r', 'sigma', 'T', 'q']].values, dtype=torch.float32)
        label = torch.tensor(self.df.iloc[idx]['label'], dtype=torch.float32)
        return input, label
    
def train(model, simulator, loss_fn=F.mse_loss, n_iters=10000, batch_size=32, lr=0.0001, device='cpu'):
    count = 0
    train_loss = []
    train_df = simulator.split_data()[0]
    option_type = simulator.option_type

    train_ds = CustomDataset(train_df)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epochs = int(n_iters / len(train_dl))

    model.train()
    for epoch in range(epochs):
        for batch in train_dl:
            input, label = batch
            optimizer.zero_grad()
            V = model(input)
            loss = loss_fn(V, label)

            loss.backward()
            optimizer.step()

            count += 1

            if count % 100 == 0:
                print(f'Epoch: {epoch}, Count: {count}, Loss: {loss.item()}')
                train_loss.append(loss.item())

    return train_loss