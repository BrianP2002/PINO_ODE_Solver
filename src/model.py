import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv1d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        self.weights = nn.Parameter(
            torch.empty(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        )
        nn.init.xavier_normal_(self.weights)

    def forward(self, x):
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x)

        out_ft = torch.zeros(batchsize, self.out_channels, x.shape[-1]//2 + 1, 
                             dtype=torch.cfloat, device=x.device)
        
        out_ft[:, :, :self.modes] = torch.einsum(
            "bix,iox->box", 
            x_ft[:, :, :self.modes], 
            self.weights
        )

        x = torch.fft.irfft(out_ft, n=x.shape[-1])
        return x

class PINO1d(nn.Module):
    def __init__(self, modes, width):
        super(PINO1d, self).__init__()
        self.modes = modes
        self.width = width
        
        # CHANGED: Input dimension is now 3 (t, x0, v0)
        self.lifting = nn.Linear(3, self.width) 
        
        self.conv0 = SpectralConv1d(self.width, self.width, self.modes)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        
        self.conv1 = SpectralConv1d(self.width, self.width, self.modes)
        self.w1 = nn.Conv1d(self.width, self.width, 1)
        
        self.projection1 = nn.Linear(self.width, 128)
        self.projection2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.lifting(x)
        x = x.permute(0, 2, 1) 
        
        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = F.gelu(x1 + x2)
        
        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = F.gelu(x1 + x2)
        
        x = x.permute(0, 2, 1) 
        x = F.gelu(self.projection1(x))
        x = self.projection2(x)
        return x