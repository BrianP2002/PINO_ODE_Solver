# Physics-Informed Neural Operator (PINO) for Non-Linear Oscillators

**Author:** Lin Ha (NetID: ZWM4367)  
**Course:** Scientific Machine Learning (ELEC_ENG / COMP_ENG 395 / 495)

## Overview
This repository contains a PyTorch implementation of a Physics-Informed Neural Operator (PINO) designed to solve the Van der Pol oscillator equations. The project demonstrates standard trajectory prediction, zero-shot super-resolution, and autoregressive temporal extrapolation.

## Repository Structure

    pino_ode_solver/
    ├── data/                  # Generated baseline data
    ├── results/               # Loss plots and trajectory visuals
    ├── src/                   # Core modules
    │   ├── __init__.py
    │   ├── model.py           # SpectralConv1d and PINO1d architecture
    │   ├── physics.py         # Finite difference PDE loss calculation
    │   └── dataset.py         # Custom PyTorch Dataset logic
    ├── generate_data.py       # Script to generate RK45 ground-truth data
    ├── train.py               # Main execution script for the training loop
    ├── evaluate.py            # Evaluation suite (standard, super-res, extrapolation)
    ├── environment.yml        # Conda environment dependencies
    └── README.md              

## Mathematical Development

### 1. The Governing Equation (Van der Pol Oscillator)
The target physical system is a non-conservative oscillator with non-linear damping. The governing second-order ordinary differential equation is:

d^2x/dt^2 - mu * (1 - x^2) * dx/dt + x = 0

where 'x' is the position coordinate and 'mu' is a scalar parameter indicating the nonlinearity and strength of the damping.

### 2. Spectral Convolution
Unlike standard Multi-Layer Perceptrons, the Neural Operator learns mappings between infinite-dimensional function spaces. The core operation is the Spectral Convolution, which performs operations in the frequency domain:
1. Transform: Apply the Fast Fourier Transform (FFT) to the spatial/temporal input.
2. Filter: Truncate to the lowest 'k' modes and multiply by a learned complex weight matrix.
3. Inverse: Apply the Inverse Fast Fourier Transform (IFFT) to return to the physical domain.

### 3. Physics-Informed Loss Function (L_pde)
To constrain the neural operator to physically valid trajectories, a physics-informed loss term is added to the standard data loss (L_data). Since the operator outputs predictions on a uniform temporal grid, derivatives are approximated using central finite differences:

First Derivative:
dx/dt ≈ (x[i+1] - x[i-1]) / (2 * dt)

Second Derivative:
d^2x/dt^2 ≈ (x[i+1] - 2x[i] + x[i-1]) / (dt^2)

The total loss optimized during training is:
L_total = L_data + lambda * L_pde

## Running the Code

### 1. Environment Setup
We use conda for environment management to ensure all dependencies are correctly handled. Run the following commands:

    conda env create -f environment.yml
    conda activate pino_env

### 2. Generate the Dataset
Generate the ground-truth training and testing data using a standard Runge-Kutta (RK45) numerical solver. This will populate the data directory with initial conditions and trajectories.

    python generate_data.py

### 3. Train the PINO Model
Train the neural operator. The script will save the best model weights to the results directory.

    python train.py

### 4. Evaluate and Visualize
Run the comprehensive evaluation suite. This script loads the trained model and automatically executes three tests, saving all plots as high-resolution PNGs:
1. Standard Evaluation: Interpolation on a 256-point grid.
2. Zero-Shot Super-Resolution: Evaluation on a 1024-point high-fidelity grid without retraining.
3. Temporal Extrapolation: Forecasting from 15s to 30s using an autoregressive rollout.

        python evaluate.py