# POD-PINO: Physics-Informed POD-DeepONet for Parameter Identification

Code for the paper:

> **"A POD-PINO Approach for Identifying Governing Parameters of a Phenomenological Model for Liquid Rocket Engine Combustion Instability"**

*(citation to be added upon publication)*

## Overview

This repository implements a proper-orthogonal-decomposition-based physics-informed neural operator (**POD-PINO**) for identifying the governing parameters (linear growth rate, nonlinear saturation coefficient, and diffusion parameter) of a stochastic nonlinear phenomenological model for combustion instability in liquid rocket engines.

The method combines **POD** (low-dimensional representation of finite-time Kramers-Moyal coefficient fields), a **DeepONet / neural operator** (parameter-to-modal-coefficient mapping), and **physics constraints** derived from the adjoint Fokker-Planck equation.

## Repository structure

- `数据集.py` — training-data generation.
- `AFP/DeepOnet/Final_code/` — adjoint Fokker-Planck solver, data generation, and DeepONet / finite-difference (FD) / KM-coefficient baselines.
- `POD_deeponet/POD_deeponet_train.py` — POD-DeepONet training.
- `POD_deeponet/PI-POD-DeepONet/` — physics-informed POD-PINO model:
  - `train.py`, `Train_*.py` — POD-PINO training.
  - `SI.py`, `SI_OUT_KM_*.py` — parameter-identification (system identification) scripts.
  - `result/` — experiment scripts (interpolation `neituicanshu/`, extrapolation `waituicanshu/`, ablation `xiaorongshiyan/`, robustness `lubangxing/`, lambda selection `Choice_lamda/`, forward prediction `forward_predict/`, stability `wendingxing/`).
- `POD_deeponet/compare/` — method-comparison scripts (FD / DeepONet / POD-DeepONet / POD-PINO).
- `POD_deeponet/plot/` — figure-generation scripts.
- `POD_deeponet/data/` — data preprocessing utilities.

## Dependencies

- Python 3.x
- PyTorch, NumPy, SciPy, pandas, scikit-learn, matplotlib, tqdm, joblib, seaborn

```bash
pip install -r requirements.txt
```

## Usage

> **Important:** the scripts contain hard-coded absolute paths (e.g. `D:\PINN\zenodo\...`). Update `DATA_DIR`, `RESULT_DIR` and other paths at the top of each script before running.

1. Generate training data with `数据集.py` and the scripts in `AFP/DeepOnet/Final_code/`.
2. Train with `POD_deeponet/POD_deeponet_train.py` (POD-DeepONet) or `POD_deeponet/PI-POD-DeepONet/Train_2.py` (POD-PINO).
3. Run identification and experiments under `POD_deeponet/PI-POD-DeepONet/result/`.
4. Generate figures with `POD_deeponet/plot/` and `POD_deeponet/compare/`.

## Data availability

Simulation datasets (`.csv`) and pre-trained model weights (`.pth`) are not included. They are available from the corresponding author upon reasonable request.

## License

MIT License - see `LICENSE`.
