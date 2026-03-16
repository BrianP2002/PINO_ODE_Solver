import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from scipy.integrate import solve_ivp
from src.model import PINO1d
from src.dataset import OscillatorDataset

seed = 20260316
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def vdp_deriv(t, y, mu=1.0):
    x, dxdt = y
    return [dxdt, mu * (1 - x**2) * dxdt - x]

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('results', exist_ok=True)
    
    model = PINO1d(modes=16, width=64).to(device)
    model_path = 'results/pino_model.pth'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}.")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    plt.subplots_adjust(hspace=0.3, wspace=0.2)

    dataset = OscillatorDataset('data/vdp_data.npz')
    features_tensor, x_true_tensor = dataset[np.random.randint(len(dataset))]
    with torch.no_grad():
        x_pred = model(features_tensor.unsqueeze(0).to(device)).cpu().squeeze().numpy()
    
    t_plot = features_tensor[:, 0].numpy()
    axes[0, 0].plot(t_plot, x_true_tensor.squeeze().numpy(), 'k--', label='Truth')
    axes[0, 0].plot(t_plot, x_pred, 'r', alpha=0.8, label='PINO')
    axes[0, 0].set_title("Standard Interpolation (256 pts)")
    axes[0, 0].legend()

    t_sr = np.linspace(0, 15, 1024)
    y0_sr = [1.0, 0.0] 
    sol_sr = solve_ivp(vdp_deriv, (0, 15), y0_sr, t_eval=t_sr)
    feat_sr = torch.cat([torch.tensor(t_sr).float().unsqueeze(-1), 
                         torch.full((1024, 1), y0_sr[0]), 
                         torch.full((1024, 1), y0_sr[1])], dim=-1).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_sr = model(feat_sr).cpu().squeeze().numpy()
    
    axes[0, 1].plot(t_sr, sol_sr.y[0], 'k--', label='Truth')
    axes[0, 1].plot(t_sr, pred_sr, 'b', alpha=0.7, label='PINO SR')
    axes[0, 1].set_title("Super-Resolution (1024 pts)")
    axes[0, 1].legend()

    t_naive = np.linspace(0, 30, 512)
    sol_naive = solve_ivp(vdp_deriv, (0, 30), y0_sr, t_eval=t_naive)
    feat_naive = torch.cat([torch.tensor(t_naive).float().unsqueeze(-1), 
                            torch.full((512, 1), y0_sr[0]), 
                            torch.full((512, 1), y0_sr[1])], dim=-1).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_naive = model(feat_naive).cpu().squeeze().numpy()
    
    axes[1, 0].plot(t_naive, sol_naive.y[0], 'k--', label='Truth')
    axes[1, 0].plot(t_naive, pred_naive, 'orange', label='Naive (Warped)')
    axes[1, 0].axvline(x=15, color='red', linestyle=':')
    axes[1, 0].set_title("Naive Extrapolation (Domain Stretching)")
    axes[1, 0].legend()

    t1 = np.linspace(0, 15, 256)
    feat1 = torch.cat([torch.tensor(t1).float().unsqueeze(-1), 
                       torch.full((256, 1), y0_sr[0]), 
                       torch.full((256, 1), y0_sr[1])], dim=-1).unsqueeze(0).to(device)
    with torch.no_grad():
        p1 = model(feat1).cpu().squeeze().numpy()
    
    x15, v15 = p1[-1], (p1[-1] - p1[-2])/(t1[1]-t1[0])
    feat2 = torch.cat([torch.tensor(t1).float().unsqueeze(-1), 
                       torch.full((256, 1), float(x15)), 
                       torch.full((256, 1), float(v15))], dim=-1).unsqueeze(0).to(device)
    with torch.no_grad():
        p2 = model(feat2).cpu().squeeze().numpy()
    
    axes[1, 1].plot(t_naive, sol_naive.y[0], 'k--', label='Truth')
    axes[1, 1].plot(t1, p1, 'r', label='Rollout 1')
    axes[1, 1].plot(t1 + 15.0, p2, 'blue', label='Rollout 2')
    axes[1, 1].axvline(x=15, color='gray', linestyle=':')
    axes[1, 1].set_title("Autoregressive Extrapolation")
    axes[1, 1].legend()

    for ax in axes.flat:
        ax.set(xlabel='Time (t)', ylabel='State (x)')
        ax.grid(True, alpha=0.3)

    plt.suptitle("PINO Van der Pol Oscillator Performance Suite", fontsize=18)
    plt.savefig('results/vdp_combined_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()