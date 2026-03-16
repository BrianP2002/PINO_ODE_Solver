# Physics-Informed Neural Operator (PINO) for Non-Linear Oscillators

**Author:** Lin Ha  
**Course:** Scientific Machine Learning (ELEC_ENG 495)

## Overview
This repository contains a PyTorch implementation of a Physics-Informed Neural Operator (PINO) designed to solve the Van der Pol oscillator equations. The project demonstrates standard trajectory prediction, zero-shot super-resolution, and autoregressive temporal extrapolation.

## Documentation
For the full mathematical derivation of the PINO architecture, the spectral convolution theory, and the physics-informed loss framework used in this project, please refer to:
* **[Mathematical_Supplement.pdf](./Mathematical_Supplement.pdf)** (Available in the root directory)

## Repository Structure

    pino_ode_solver/
    ├── data/                  # Generated baseline data
    ├── results/               # Loss plots and trajectory visuals
    ├── src/                   # Core modules
    │   ├── model.py           # SpectralConv1d and PINO1d architecture
    │   ├── physics.py         # Finite difference PDE loss calculation
    │   └── dataset.py         # Custom PyTorch Dataset logic
    ├── generate_data.py       # Script to generate RK45 ground-truth data
    ├── train.py               # Main execution script for the training loop
    ├── evaluate.py            # Evaluation suite (standard, super-res, extrapolation)
    ├── environment.yml        # Conda environment dependencies
    └── README.md              

## Running the Code

### 1. Environment Setup
    conda env create -f environment.yml
    conda activate pino_env

### 2. Generate the Dataset
    python generate_data.py

### 3. Train the PINO Model
    python train.py

### 4. Evaluate and Visualize
    python evaluate.py