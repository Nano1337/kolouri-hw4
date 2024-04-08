import pickle
import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, TensorDataset

from tqdm import tqdm
import matplotlib.pyplot as plt

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 2)
        )
    
    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

class SimpleGan(nn.Module):
    def __init__(self):
        super(SimpleGan, self).__init__()
        self.generator = Generator()
        self.discriminator = Discriminator()

    def forward(self, x):
        pass
    
class GANLoss(nn.Module):
    def __init__(self):
        super(GANLoss, self).__init__()
        self.loss = nn.BCELoss()
    
    def forward(self, x, y):
        return self.loss(x, y)
    
def train_gan(data, epochs=100, batch_size=64):
    # Convert data to tensor
    data = torch.tensor(data, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # define model, loss, optimizer
    gan = SimpleGan().to(device)
    criterion = GANLoss().to(device)
    optimizer_g = torch.optim.Adam(gan.generator.parameters(), lr=0.0001) 
    optimizer_d = torch.optim.Adam(gan.discriminator.parameters(), lr=0.0001)
    dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)
    
    for epoch in tqdm(range(epochs), desc="Epochs"):
        d_losses = []
        g_losses = []
        real_accs = []
        fake_accs = []
        for i, real_samples in enumerate(dataloader):
            real_samples = real_samples.to(device)
            
            # Train Discriminator for 10 iterations
            for _ in range(10):
                optimizer_d.zero_grad()
                
                # Real samples
                real_preds = gan.discriminator(real_samples)
                real_targets = torch.ones_like(real_preds).to(device)
                real_loss = criterion(real_preds, real_targets)
                real_acc = (real_preds > 0.5).float().mean()
                
                # Fake samples
                latent = torch.randn_like(real_samples).to(device)
                fake_samples = gan.generator(latent)
                fake_preds = gan.discriminator(fake_samples)
                fake_targets = torch.zeros_like(fake_preds).to(device)
                fake_loss = criterion(fake_preds, fake_targets)
                fake_acc = (fake_preds < 0.5).float().mean()
                
                # Combine losses
                d_loss = real_loss + fake_loss
                d_loss.backward()
                optimizer_d.step()
                
                d_losses.append(d_loss.item())
                real_accs.append(real_acc.item())
                fake_accs.append(fake_acc.item())

            # Train Generator for 3 iterations
            for _ in range(3):
                optimizer_g.zero_grad()
                latent = torch.randn_like(real_samples).to(device)
                fake_samples = gan.generator(latent)
                preds = gan.discriminator(fake_samples)
                targets = torch.ones_like(preds).to(device)
                g_loss = criterion(preds, targets)
                g_loss.backward()
                optimizer_g.step()
                
                g_losses.append(g_loss.item())
        
        avg_d_loss = sum(d_losses) / len(d_losses)
        avg_g_loss = sum(g_losses) / len(g_losses)
        avg_real_acc = sum(real_accs) / len(real_accs)
        avg_fake_acc = sum(fake_accs) / len(fake_accs)
        
        tqdm.write(f"Epoch {epoch+1}/{epochs}, Loss D: {avg_d_loss:.4f}, Loss G: {avg_g_loss:.4f}, Real Acc: {avg_real_acc:.4f}, Fake Acc: {avg_fake_acc:.4f}")
        
    return gan

if __name__ == "__main__":

    # train model
    with open("hw4_p1.pkl", "rb") as f:
        data = pickle.load(f) # (2000, 2) shape
        model = train_gan(data)

    # first figure
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X,Y=torch.meshgrid(torch.linspace(0.0,8.0, 100),torch.linspace(0.0,8.0, 100))
    xgrid=torch.stack([X.reshape(-1),Y.reshape(-1)],1)
    discGrid=model.discriminator(xgrid.to(device))
    discGrid=discGrid.detach().cpu().numpy()
    plt.scatter(xgrid[:,0],xgrid[:,1],c=discGrid)
    plt.savefig("prob1_1.png")

    # second figure
    z=torch.randn((2000,2))
    xhat=model.generator(z.to(device))
    xhat=xhat.detach().cpu()
    fig,ax=plt.subplots(1,2,figsize=(10,5))
    ax[0].scatter(z[:,0],z[:,1])
    ax[0].set_title("Network's input")
    ax[1].scatter(xhat[:,0],xhat[:,1])
    ax[1].set_title("Network's output")
    plt.savefig("prob1_2.png")


    

    

    