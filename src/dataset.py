import torch
from torch.utils.data import Dataset
import numpy as np

class OscillatorDataset(Dataset):
    def __init__(self, data_path):
        data = np.load(data_path)
        self.t = torch.tensor(data['t'], dtype=torch.float32).unsqueeze(-1)
        self.x = torch.tensor(data['x'], dtype=torch.float32).unsqueeze(-1)
        self.v = torch.tensor(data['v'], dtype=torch.float32).unsqueeze(-1)
        
    def __len__(self):
        return self.x.shape[0]
        
    def __getitem__(self, idx):
        t_val = self.t[idx]
        x_val = self.x[idx]
        v_val = self.v[idx]
        
        # Extract initial conditions and expand them to match the time grid
        x0 = x_val[0].expand_as(t_val)
        v0 = v_val[0].expand_as(t_val)
        
        # Input features: [Time, Initial Position, Initial Velocity]
        features = torch.cat([t_val, x0, v0], dim=-1)
        
        return features, x_val