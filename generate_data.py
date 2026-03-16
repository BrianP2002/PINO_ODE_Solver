import numpy as np
from scipy.integrate import solve_ivp
import os
import torch
import numpy as np
import random

seed = 20260316
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

def vdp_deriv(t, y, mu):
    x, dxdt = y
    return [dxdt, mu * (1 - x**2) * dxdt - x]

def generate_data(num_samples=200, t_span=(0, 15), num_points=256, mu=1.0):
    t_eval = np.linspace(t_span[0], t_span[1], num_points)
    all_x = []
    all_v = []
    all_t = []
    
    for _ in range(num_samples):
        y0 = [np.random.uniform(-2, 2), np.random.uniform(-2, 2)]
        sol = solve_ivp(vdp_deriv, t_span, y0, args=(mu,), t_eval=t_eval)
        all_x.append(sol.y[0])
        all_v.append(sol.y[1])  # Save velocity 
        all_t.append(t_eval)
        
    os.makedirs('data', exist_ok=True)
    np.savez('data/vdp_data.npz', t=np.array(all_t), x=np.array(all_x), v=np.array(all_v))
    print("Data generated and saved to data/vdp_data.npz")

if __name__ == "__main__":
    generate_data()