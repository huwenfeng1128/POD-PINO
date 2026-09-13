# POD-PINO: Physics-Informed POD-DeepONet for Parameter Identification

Code for the paper:

> **"A POD-PINO Approach for Identifying Governing Parameters of a Phenomenological Model for Liquid Rocket Engine Combustion Instability"**

*(citation to be added upon publication)*

## Overview

This repository implements a proper-orthogonal-decomposition-based physics-informed neural operator (**POD-PINO**) for identifying the governing parameters (linear growth rate, nonlinear saturation coefficient, and diffusion parameter) of a stochastic nonlinear phenomenological model for combustion instability in liquid rocket engines.

The method combines:

- **Proper Orthogonal Decomposition (POD)** for a low-dimensional representation of the finite-time Kramers-Moyal (KM) coefficient fields.
- **DeepONet / neural operator** for learning the parameter-to-modal-coefficient mapping.
- **Physics constraints** derived from the adjoint Fokker-Planck equation (residuals of the governing equations for the finite-time KM coefficients) embedded in the training loss.

## Repository structure

- `AFP/` — adjoint Fokker-Planck solver, data generation, and baseline models (finite-difference identification, DeepONet, POD-DeepONet).
- `POD_deeponet/` — POD-DeepONet and the physics-informed POD-PINO model (training, evaluation, plotting).
  - `POD_deeponet/PI-POD-DeepONet/` — physics-informed POD-PINO training scripts.
  - `POD_deeponet/compare/`, `plot/`, `data/` — comparison and plotting utilities.

> Note: the repository retains the original development scripts (including experimental variants) for reproducibility. Key training entry points:
> - `POD_deeponet/POD_deeponet_train.py` — POD-DeepONet training.
> - `POD_deeponet/PI-POD-DeepONet/Train_2.py` — physics-informed POD-PINO training.

## Dependencies

- Python 3.x
- PyTorch, NumPy, SciPy, pandas, scikit-learn, matplotlib, tqdm, joblib

```bash
pip install -r requirements.txt
```

## Usage

> **Important:** the original scripts contain hard-coded absolute paths (e.g. `D:\PINN\zenodo\...`). Before running, update `DATA_DIR`, `RESULT_DIR`, and other paths at the top of each script to match your local setup.

1. Generate the training data (adjoint Fokker-Planck solutions) using the data-generation scripts in `AFP/`.
2. Train with `POD_deeponet/POD_deeponet_train.py` or `POD_deeponet/PI-POD-DeepONet/Train_2.py`.
3. Evaluate and plot with scripts in `POD_deeponet/compare/` and `POD_deeponet/plot/`.

## Data availability

Simulation datasets (`.csv`) and pre-trained model weights (`.pth`) are excluded from this repository. They are available from the corresponding author upon reasonable request.

## License

MIT License - see `LICENSE`.
