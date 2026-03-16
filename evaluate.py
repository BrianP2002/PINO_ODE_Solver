import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.integrate import solve_ivp
from src.model import PINO1d
from src.dataset import OscillatorDataset

# Helper function for the Van der Pol ODE
def vdp_deriv(t, y, mu=1.0):
    x, dxdt = y
    return [dxdt, mu * (1 - x**2) * dxdt - x]

def eval_standard(model, device):
    print("1. Running Standard Evaluation...")
    dataset = OscillatorDataset('data/vdp_data.npz')
    sample_idx = np.random.randint(len(dataset))
    features_tensor, x_true_tensor = dataset[sample_idx]
    
    features_input = features_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        x_pred_tensor = model(features_input)
        
    t_plot = features_tensor[:, 0].numpy()
    x_true_plot = x_true_tensor.squeeze().numpy()
    x_pred_plot = x_pred_tensor.cpu().squeeze().numpy()
    
    plt.figure(figsize=(10, 5))
    plt.plot(t_plot, x_true_plot, label='Ground Truth (RK45)', color='black', linewidth=2, linestyle='dashed')
    plt.plot(t_plot, x_pred_plot, label='PINO Prediction', color='red', alpha=0.8, linewidth=2)
    
    plt.title('Van der Pol Oscillator: PINO vs Numerical Solver', fontsize=14)
    plt.xlabel('Time (t)', fontsize=12)
    plt.ylabel('State (x)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    save_path = 'results/vdp_evaluation.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   -> Saved: {save_path}")

def eval_super_resolution(model, device):
    print("2. Running Zero-Shot Super-Resolution...")
    t_span = (0, 15)
    num_points = 1024 
    t_eval = np.linspace(t_span[0], t_span[1], num_points)
    
    y0 = [np.random.uniform(-2, 2), np.random.uniform(-2, 2)]
    sol = solve_ivp(vdp_deriv, t_span, y0, t_eval=t_eval)
    
    t_tensor = torch.tensor(t_eval, dtype=torch.float32).unsqueeze(-1)
    x0_tensor = torch.full_like(t_tensor, y0[0])
    v0_tensor = torch.full_like(t_tensor, y0[1])
    
    features = torch.cat([t_tensor, x0_tensor, v0_tensor], dim=-1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_tensor = model(features)
        
    x_pred = pred_tensor.cpu().squeeze().numpy()
    x_true = sol.y[0]

    plt.figure(figsize=(10, 5))
    plt.plot(t_eval, x_true, label='Ground Truth (1024 pts)', color='black', linewidth=2, linestyle='dashed')
    plt.plot(t_eval, x_pred, label='PINO Zero-Shot Output', color='blue', alpha=0.7, linewidth=2)
    
    plt.title('Zero-Shot Super-Resolution (Trained on 256 pts, Evaluated on 1024 pts)', fontsize=14)
    plt.xlabel('Time (t)', fontsize=12)
    plt.ylabel('State (x)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    save_path = 'results/vdp_super_resolution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   -> Saved: {save_path}")

def eval_extrapolation_naive(model, device):
    print("3. Running Naive Temporal Extrapolation (Demonstrates FNO failure mode)...")
    t_span = (0, 30)
    num_points = 512 
    t_eval = np.linspace(t_span[0], t_span[1], num_points)
    
    y0 = [np.random.uniform(-2, 2), np.random.uniform(-2, 2)]
    sol = solve_ivp(vdp_deriv, t_span, y0, t_eval=t_eval)
    
    t_tensor = torch.tensor(t_eval, dtype=torch.float32).unsqueeze(-1)
    x0_tensor = torch.full_like(t_tensor, y0[0])
    v0_tensor = torch.full_like(t_tensor, y0[1])
    
    features = torch.cat([t_tensor, x0_tensor, v0_tensor], dim=-1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_tensor = model(features)
        
    x_pred = pred_tensor.cpu().squeeze().numpy()
    x_true = sol.y[0]

    plt.figure(figsize=(12, 5))
    plt.plot(t_eval, x_true, label='Ground Truth (RK45)', color='black', linewidth=2, linestyle='dashed')
    plt.plot(t_eval, x_pred, label='PINO Prediction (Warped)', color='orange', alpha=0.8, linewidth=2)
    
    plt.axvline(x=15.0, color='red', linestyle=':', linewidth=2, label='End of Training Domain')
    
    plt.title('Naive Extrapolation: Stretching the Fourier Domain', fontsize=14)
    plt.xlabel('Time (t)', fontsize=12)
    plt.ylabel('State (x)', fontsize=12)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3)
    
    save_path = 'results/vdp_extrapolation_naive.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   -> Saved: {save_path}")

def eval_extrapolation_autoregressive(model, device):
    print("4. Running Autoregressive Extrapolation (Proper Method)...")
    # Ground truth for full 30 seconds
    t_span_full = (0, 30)
    t_eval_full = np.linspace(t_span_full[0], t_span_full[1], 512)
    y0 = [np.random.uniform(-2, 2), np.random.uniform(-2, 2)]
    sol = solve_ivp(vdp_deriv, t_span_full, y0, t_eval=t_eval_full)
    x_true = sol.y[0]
    
    # --- Rollout 1 (0 to 15s) ---
    t_eval_1 = np.linspace(0, 15, 256)
    dt = t_eval_1[1] - t_eval_1[0]
    
    t_tensor_1 = torch.tensor(t_eval_1, dtype=torch.float32).unsqueeze(-1)
    x0_tensor_1 = torch.full_like(t_tensor_1, y0[0])
    v0_tensor_1 = torch.full_like(t_tensor_1, y0[1])
    
    features_1 = torch.cat([t_tensor_1, x0_tensor_1, v0_tensor_1], dim=-1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_1 = model(features_1).cpu().squeeze().numpy()
        
    # --- Rollout 2 (15 to 30s) ---
    # Extract boundary conditions at t=15
    x15 = pred_1[-1]
    # Estimate velocity using backward finite difference: v = dx/dt
    v15 = (pred_1[-1] - pred_1[-2]) / dt 
    
    # Re-use the 0-15s grid for the model, but with new initial conditions
    x0_tensor_2 = torch.full_like(t_tensor_1, float(x15))
    v0_tensor_2 = torch.full_like(t_tensor_1, float(v15))
    features_2 = torch.cat([t_tensor_1, x0_tensor_2, v0_tensor_2], dim=-1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_2 = model(features_2).cpu().squeeze().numpy()
        
    # --- Plotting ---
    plt.figure(figsize=(12, 5))
    plt.plot(t_eval_full, x_true, label='Ground Truth (RK45)', color='black', linewidth=2, linestyle='dashed')
    
    # Plot Rollout 1 exactly as is
    plt.plot(t_eval_1, pred_1, label='PINO Rollout 1 (0-15s)', color='red', linewidth=2)
    
    # Shift the x-axis for Rollout 2 by adding 15 seconds to the time array
    plt.plot(t_eval_1[1:] + 15.0, pred_2[1:], label='PINO Rollout 2 (15-30s)', color='blue', linewidth=2)
    
    plt.axvline(x=15.0, color='gray', linestyle=':', linewidth=2, label='Rollout Boundary')
    
    plt.title('Autoregressive Extrapolation: Forecasting by Stitching Domains', fontsize=14)
    plt.xlabel('Time (t)', fontsize=12)
    plt.ylabel('State (x)', fontsize=12)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3)
    
    save_path = 'results/vdp_extrapolation_autoregressive.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"   -> Saved: {save_path}")

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Initializing evaluations on: {device}")
    
    os.makedirs('results', exist_ok=True)
    
    # Initialize and load model once
    model = PINO1d(modes=16, width=64).to(device)
    model_path = 'results/pino_model.pth'
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}. Run train.py first.")
        
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    # Run all evaluations
    eval_standard(model, device)
    eval_super_resolution(model, device)
    eval_extrapolation_naive(model, device)
    eval_extrapolation_autoregressive(model, device)
    
    print("All evaluations complete. Displaying plots...")
    plt.show()

if __name__ == "__main__":
    main()