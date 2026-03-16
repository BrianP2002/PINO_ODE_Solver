import torch
import torch.nn.functional as F

def pino_loss(model_output, target_data, grid_spacing, mu=1.0, lambda_pde=0.1):
    data_loss = F.mse_loss(model_output, target_data)
    
    x = model_output.squeeze(-1)
    
    dx_dt = (torch.roll(x, shifts=-1, dims=1) - torch.roll(x, shifts=1, dims=1)) / (2 * grid_spacing)
    d2x_dt2 = (torch.roll(x, shifts=-1, dims=1) - 2 * x + torch.roll(x, shifts=1, dims=1)) / (grid_spacing ** 2)
    
    x_inner = x[:, 1:-1]
    dx_dt_inner = dx_dt[:, 1:-1]
    d2x_dt2_inner = d2x_dt2[:, 1:-1]
    
    physics_residual = d2x_dt2_inner - mu * (1 - x_inner**2) * dx_dt_inner + x_inner 
    pde_loss = torch.mean(physics_residual ** 2)
    
    total_loss = data_loss + (lambda_pde * pde_loss)
    return total_loss, data_loss, pde_loss