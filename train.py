import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.model import PINO1d
from src.physics import pino_loss
from src.dataset import OscillatorDataset
import os

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    
    dataset = OscillatorDataset('data/vdp_data.npz')
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    modes = 16
    width = 64
    model = PINO1d(modes, width).to(device)
    
    # Added a StepLR scheduler to help it settle into the minima faster
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    epochs = 150
    grid_spacing = dataset.t[0, 1, 0].item() - dataset.t[0, 0, 0].item()
    
    os.makedirs('results', exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        
        for features, x in dataloader:
            features, x = features.to(device), x.to(device)
            
            optimizer.zero_grad()
            pred = model(features)
            
            loss, data_loss, pde_loss = pino_loss(pred, x, grid_spacing)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        scheduler.step()
            
        if epoch % 10 == 0:
            avg_loss = total_train_loss / len(dataloader)
            print(f'Epoch {epoch}: Total Loss = {avg_loss:.4f}, Data = {data_loss.item():.4f}, PDE = {pde_loss.item():.4f}')
            
    torch.save(model.state_dict(), 'results/pino_model.pth')
    print("Training complete. Model saved to results/pino_model.pth")

if __name__ == "__main__":
    train()