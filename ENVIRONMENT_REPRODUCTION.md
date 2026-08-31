# Reproduction environment

Validated locally on Windows with Python 3.11, PyTorch 2.6.0+cu124, CUDA,
XGBoost 3.2.0, scikit-learn 1.8.0, and SciPy 1.17.1. The formal configuration
targets an NVIDIA GPU; classical models and the HiGHS MILP solver run on CPU.

The source paper reports PyTorch 2.1.1 on an NVIDIA TITAN GPU but does not state
the exact TITAN model, CUDA/cuDNN versions, random seed, deterministic-kernel
settings, or package versions for QRGBM and optimization. Exact floating-point
identity is therefore not a defensible reproduction target. The experiment
archives retain configuration, split seed, checkpoint path, sampler, denoising
steps, and scenario count so numerical deviations can be audited.

The formal training configuration uses fixed 26,000 updates and intentionally
does not early-stop: this follows the only training-duration rule disclosed by
the paper. Checkpoints are written every 1,000 updates and training resumes from
`latest.pt` after interruption.
