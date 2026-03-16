# Physics-Informed Neural Operator (PINO) for Non-Linear Oscillators

**Author:** Lin Ha
**Course:** Scientific Machine Learning (ELEC_ENG 495)

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

$$\frac{d^2x}{dt^2} - \mu(1 - x^2)\frac{dx}{dt} + x = 0$$

where $x$ is the position coordinate and $\mu$ is a scalar parameter indicating the nonlinearity and strength of the damping.

### 2. Spectral Convolution
The core operation of the Neural Operator is the **Spectral Convolution**, which performs operations in the frequency domain to learn mappings between infinite-dimensional function spaces:

1. **Transform**: Apply the Fast Fourier Transform (FFT) to the temporal input:
<p align="center">
<img src="https://render.githubusercontent.com/render/math?math=%5Chat%7Bx%7D%20%3D%20%5Cmathcal%7BF%7D(x)">
</p>

$$\hat{x} = \mathcal{F}(x)$$

2. **Filter**: Truncate to the lowest $k$ modes and multiply by learned complex weights $R$:
$$\hat{x}_{out} = R \cdot \hat{x}_{in}$$

3. **Inverse**: Apply the Inverse FFT to return to the physical domain:
$$x_{out} = \mathcal{F}^{-1}(\hat{x}_{out})$$

### 3. Physics-Informed Loss Function ($\mathcal{L}_{pde}$)
To ensure physical consistency, a physics-informed loss term is added to the data loss ($\mathcal{L}_{data}$). Derivatives are approximated using central finite differences on the uniform grid:

**First Derivative:**
$$\frac{dx}{dt} \approx \frac{x_{i+1} - x_{i-1}}{2\Delta t}$$

**Second Derivative:**
$$\frac{d^2x}{dt^2} \approx \frac{x_{i+1} - 2x_i + x_{i-1}}{\Delta t^2}$$

**Total Loss:**
$$\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda \mathcal{L}_{pde}$$

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