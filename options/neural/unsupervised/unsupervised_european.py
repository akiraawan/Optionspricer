import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from options.neural.simulator import SimulatorEuropean


class European_Unsup_NN(nn.Module):
    def __init__(self):
        super(European_Unsup_NN, self).__init__()
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

    def forward(self, S, K, r, sigma, T, q):
        S, K, r, sigma, T, q = S.unsqueeze(1), K.unsqueeze(1), r.unsqueeze(1), sigma.unsqueeze(1), T.unsqueeze(1), q.unsqueeze(1)
        inputs = torch.cat([S, K, r, sigma, T, q], dim=1)  # bsx6
        # inputs is type df
        outputs = self.seq(inputs).squeeze(1).requires_grad_()
        return outputs


def loss_fn(V, S, K, r, sigma, T, q, type='call'):
    L_PDE = 0.0
    L_BC = 0.0

    dVdT = torch.autograd.grad(V, T, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    dVdS = torch.autograd.grad(V, S, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    d2VdS2 = torch.autograd.grad(dVdS, S, grad_outputs=torch.ones_like(V), create_graph=True)[0]

    indices_Tg0 = np.where(T > 0)[0]

    S_Tg0 = S[indices_Tg0]
    K_Tg0 = K[indices_Tg0]
    r_Tg0 = r[indices_Tg0]
    sigma_Tg0 = sigma[indices_Tg0]
    q_Tg0 = q[indices_Tg0]
    dVdT_Tg0 = dVdT[indices_Tg0]
    dVdS_Tg0 = dVdS[indices_Tg0]
    d2VdS2_Tg0 = d2VdS2[indices_Tg0]
    V_Tg0 = V[indices_Tg0]

    L_PDE += torch.mean(torch.square(-dVdT_Tg0 + (r_Tg0 - q_Tg0) * S_Tg0 * dVdS_Tg0 + 0.5 * sigma_Tg0 ** 2 * S_Tg0 ** 2 * d2VdS2_Tg0 - r_Tg0 * V_Tg0))  # dv/dt + (r-q)s dv/ds + 0.5 sigma^2 s^2 d2v/ds2 - rV = 0

    if type == 'call':
        L_BC += torch.mean(torch.square(V - torch.max(S - K * torch.exp(-r * T), torch.zeros_like(S))))
    else:
        L_BC += torch.mean(torch.square(V - torch.max(K * torch.exp(-r * T) - S, torch.zeros_like(S))))

    return L_PDE + L_BC


class CustomDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        S = torch.tensor(self.df.iloc[idx]['S'], dtype=torch.float32, requires_grad=True)
        K = torch.tensor(self.df.iloc[idx]['K'], dtype=torch.float32, requires_grad=True)
        r = torch.tensor(self.df.iloc[idx]['r'], dtype=torch.float32, requires_grad=True)
        sigma = torch.tensor(self.df.iloc[idx]['sigma'], dtype=torch.float32, requires_grad=True)
        T = torch.tensor(self.df.iloc[idx]['T'], dtype=torch.float32, requires_grad=True)
        q = torch.tensor(self.df.iloc[idx]['q'], dtype=torch.float32, requires_grad=True)

        return S, K, r, sigma, T, q


def train(model, simulator, n_iters=10000, batch_size=32, lr=0.0001, device='cpu'):
    count = 0
    train_loss = []
    train_df = simulator.df  # For unsupervised, train data on whole dataset
    option_type = simulator.option_type

    train_ds = CustomDataset(train_df)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epochs = int(n_iters / len(train_dl))

    model.train()
    for epoch in range(epochs):
        for batch in train_dl:
            # S, K, r, sigma, T, q = [param.to(device) for param in batch]
            S, K, r, sigma, T, q = batch
            optimizer.zero_grad()
            V = model(S, K, r, sigma, T, q)
            loss = loss_fn(V, S, K, r, sigma, T, q, option_type)

            loss.backward()
            optimizer.step()

            count += 1

            if count % 100 == 0:
                print(f'Epoch: {epoch}, Count: {count}, Loss: {loss.item()}')
                train_loss.append(loss.item())

    return train_loss