import torch
import torch.nn as nn
import numpy as np
import pickle
import logging
from tqdm import tqdm
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO)

class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: torch.Tensor):
        x = x * self.scale
        half_size = self.size // 2
        emb = torch.log(torch.tensor([10000.0], device=device)) / (half_size - 1)
        emb = torch.exp(-emb * torch.arange(half_size, device=device).float())
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=-1)
        return emb

class NoisePredictor(nn.Module):
    def __init__(self, embedding_size=128):
        super(NoisePredictor, self).__init__()
        self.time_embedding = SinusoidalEmbedding(embedding_size)
        self.model = nn.Sequential(
            nn.Linear(2 + embedding_size, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 2)
        )

    def forward(self, x, t):
        t_emb = self.time_embedding(t.float()).squeeze(1)
        x = torch.cat([x, t_emb], dim=-1)
        return self.model(x)

# Load dataset and move to device
with open("hw4_p1.pkl", "rb") as f:
    data = pickle.load(f)

dataset = data
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Hyperparameters and tensors to CUDA
T = 500
beta = torch.linspace(1e-3, 1e-1, T, device=device)**2
alpha = 1 - beta
alpha_bar = torch.cumprod(alpha, dim=0)

num_epochs = 1000

# Initialize model and optimizer
model = NoisePredictor().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
    epoch_loss = 0.0
    for x0 in dataloader:
        x0 = x0.to(device)
        t = torch.randint(0, T, (x0.size(0),), device=device)
        noise = torch.randn_like(x0)

        alpha_bar_t = alpha_bar[t].unsqueeze(-1)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
        xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

        predicted_noise = model(xt, t)
        loss = nn.MSELoss()(predicted_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(dataloader)
    logging.info(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_epoch_loss:.4f}")

# Make plots
batch_size = 10000
fig = plt.figure(figsize=(20, 8), constrained_layout=True)

# Create 2x1 subfigures
subfigs = fig.subfigures(nrows=2, ncols=1)
titles = ["Forward Process", "Reverse Process"]
plot_intervals = np.linspace(1, T, 7, dtype=int)
intervals = [0, 50, 100, 150, 200, 250, 500]

for row, subfig in enumerate(subfigs):
    subfig.suptitle(titles[row], fontweight='bold')
    axs = subfig.subplots(1, 7)  # Create 1x7 grid of axes in each subfigure

    if row == 0:  # Forward Process
        with torch.no_grad():
            x0 = data.to(device)
            for t in tqdm(plot_intervals, desc="Forward Noising Steps"):
                noise = torch.randn_like(x0)
                alpha_bar_t = alpha_bar[t-1].unsqueeze(-1)
                sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
                sqrt_one_minus_alpha_bar_t = torch.sqrt(1 - alpha_bar_t)
                xt = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

                ax_idx = np.where(plot_intervals == t)[0][0]
                axs[ax_idx].scatter(xt[:, 0].cpu(), xt[:, 1].cpu(), s=2)
                axs[ax_idx].set_xlim([-10, 10])
                axs[ax_idx].set_ylim([-10, 10])
                axs[ax_idx].set_title(f"t={intervals[ax_idx]}")
    else:  # Reverse Process Inference
        with torch.no_grad():
            x = torch.randn((batch_size, 2), device=device)
            for i, t in enumerate(tqdm(reversed(range(1, T+1)), desc="Inference Steps")):
                z = torch.randn_like(x) if t > 1 else torch.zeros_like(x)
                alpha_t = alpha[t-1]
                alpha_bar_t = alpha_bar[t-1]
                sigma_t = torch.sqrt((1 - alpha_bar_t) / (1 - alpha_bar[t-2]) * beta[t-1]) if t > 1 else torch.sqrt(beta[t-1])

                predicted_noise = model(x, torch.full((batch_size,), t-1, device=device))
                x = 1 / torch.sqrt(alpha_t) * (x - (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t) * predicted_noise) + sigma_t * z

                if t in plot_intervals:
                    ax_idx = len(plot_intervals) - np.where(plot_intervals == t)[0][0] - 1
                    axs[ax_idx].scatter(x[:, 0].cpu(), x[:, 1].cpu(), s=1)
                    axs[ax_idx].set_xlim([-10, 10])
                    axs[ax_idx].set_ylim([-10, 10])
                    axs[ax_idx].set_title(f"t={intervals[ax_idx]}")

plt.savefig("prob2.png")