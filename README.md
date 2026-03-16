# Physics-Informed Neural Operator (PINO) for Non-Linear Oscillators

**Author:** Lin Ha  
**Course:** Scientific Machine Learning (ELEC_ENG 495)

This repository contains a PyTorch implementation of a Physics-Informed Neural Operator (PINO) designed to solve the Van der Pol oscillator equations. The project demonstrates standard trajectory prediction, zero-shot super-resolution, and autoregressive temporal extrapolation.

## 1. Environment Setup
We use `conda` for environment management to ensure all dependencies are correctly handled.

    conda env create -f environment.yml
    conda activate pino_env

## 2. Generate the Dataset
Generate the ground-truth training and testing data using a standard Runge-Kutta (RK45) numerical solver. This will populate the `data/` directory with initial conditions and trajectories.

    python generate_data.py

## 3. Train the PINO Model
Train the neural operator using a combined loss function that penalizes both data mismatch and violations of the governing physical equations. The trained weights will be saved to the `results/` directory.

    python train.py

## 4. Evaluate and Visualize
Run the comprehensive evaluation suite. This script will load the trained model and automatically execute three tests, saving all plots as high-resolution PNGs in the `results/` folder:

1. **Standard Evaluation:** Interpolation on the 256-point grid.
2. **Zero-Shot Super-Resolution:** Evaluation on a 1024-point high-fidelity grid without retraining.
3. **Temporal Extrapolation:** Forecasting from 15s to 30s using an autoregressive rollout.

        python evaluate.py