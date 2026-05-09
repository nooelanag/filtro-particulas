# Particle Filter vs Kalman Filter — Bayesian State Estimation

A Python implementation of **Sequential Importance Sampling (SIS) with systematic resampling** and **Kalman filtering** for tracking a drifting ship using noisy sonar observations. The project provides a complete experimental framework to compare both estimators under varying noise conditions, non-Gaussian disturbances, and computational constraints.

---

## Problem Statement

A ship drifts according to a linear state-space model driven by ocean currents:

```
x_t = A x_{t-1} + u_t       (state transition)
y_t = B x_t + v_t           (sonar observation)
```

| Symbol | Description | Value |
|--------|-------------|-------|
| `x_t` | 2D position vector | — |
| `A` | Transition matrix | `[[0.9, 0.12], [0.08, 0.85]]` |
| `B` | Observation matrix | `[[-0.12, 0.15], [0.4, -0.15]]` |
| `u_t` | System noise | `N(0, σ²_u I₂)` |
| `v_t` | Observation noise | `N(0, σ²_v I₂)` |
| `x_0` | Prior | `N(0, 25 I₂)` |

The position `x_t` is never observed directly — only the noisy sonar readings `y_t` are available.

---

## Features

- **System simulator** with configurable Gaussian or uniform noise, supporting Monte Carlo experiments over independent realizations.
- **Kalman filter** — optimal linear estimator with full predict-update recursion.
- **Particle filter (SIS)** — prior as importance distribution, log-domain weight computation for numerical stability, and systematic resampling triggered by effective sample size.
- **Uniform-noise particle filter** — indicator likelihood adapted to bounded support distributions, with distance-based fallback when no particles fall inside the support.
- **Direct estimation** via `B⁻¹ y` as a naive baseline.
- **Posterior functional estimation** — variance of `z₁₀₀ = ‖x₁₀₀‖` conditioned on all observations, computed from weighted particles.
- **Automated experiments** — generates all figures for the five required analyses in a single run.

---

## Repository Structure

```
.
├── src/
│   └── particle_filter_lab.py    # Full implementation and experiments
├── .gitignore
└── README.md
```

---

## Requirements

| Package | Version | Purpose |
|---------|---------|---------|
| Python | ≥ 3.8 | Runtime |
| NumPy | ≥ 1.20 | Linear algebra, random sampling |
| Matplotlib | ≥ 3.4 | Figure generation |

No external frameworks or compiled dependencies are needed.

### Installation

```bash
# Clone the repository
git clone https://github.com/nooelanag/filtro-particulas.git
cd particle-filter-lab

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
venv\Scripts\activate      # Windows

# Install dependencies
pip install numpy matplotlib
```

---

## Usage

Run the full experiment suite:

```bash
python src/particle_filter_lab.py
```

This executes all five sections sequentially and writes PNG figures to the working directory:

| Output file | Experiment |
|-------------|------------|
| `fig1_trayectoria_estimada.png` | Real vs estimated trajectory (both filters) |
| `fig2_efecto_sigma.png` | Effect of σ_u / σ_v ratio on accuracy |
| `fig3_mse_comparacion.png` | MSE over 50 simulations (Kalman, PF, direct) |
| `fig4_mse_uniforme.png` | MSE under uniform noise (model mismatch) |
| `fig5_tiempos_ejecucion.png` | Execution time comparison |
| `fig6_varianza_z100.png` | Posterior distribution of z₁₀₀ = ‖x₁₀₀‖ |

Numerical results (MSE values, timing, variance estimates) are printed to stdout.

### Using Individual Components

Each filter is implemented as a standalone function and can be imported independently:

```python
from src.particle_filter_lab import (
    simulate_system,
    kalman_filter,
    particle_filter,
    particle_filter_uniform,
    direct_estimation,
    estimate_variance_z100,
    A, B,
)

# Generate a trajectory
x_true, y_obs = simulate_system(A, B, sigma_u=1.0, sigma_v=1.0, T=100)

# Run Kalman filter
x_kalman, P_history = kalman_filter(y_obs, A, B, sigma_u=1.0, sigma_v=1.0)

# Run particle filter with 1000 particles
x_pf, particles, weights = particle_filter(y_obs, A, B, 1.0, 1.0, N_particles=1000)

# Estimate Var(z_100 | y_{1:100})
var_z, mean_z, z_particles, w = estimate_variance_z100(y_obs, A, B, 1.0, 1.0, 5000)
```

---

## Architecture

### Module API

| Function | Inputs | Returns | Description |
|----------|--------|---------|-------------|
| `simulate_system` | `A, B, σ_u, σ_v, T, noise_type` | `x (2×T+1), y (2×T)` | State trajectory and observations |
| `kalman_filter` | `y, A, B, σ_u, σ_v` | `x_est (2×T+1), P_hist` | Filtered estimates and covariances |
| `particle_filter` | `y, A, B, σ_u, σ_v, N` | `x_est, particles, weights` | SIS with systematic resampling |
| `particle_filter_uniform` | `y, A, B, σ_u, σ_v, N` | `x_est, particles, weights` | PF with indicator likelihood |
| `direct_estimation` | `y, B` | `x_est (2×T)` | Naive B⁻¹y inversion |
| `estimate_variance_z100` | `y, A, B, σ_u, σ_v, N` | `var, mean, z_particles, weights` | Posterior variance of ‖x₁₀₀‖ |
| `systematic_resampling` | `weights` | `indices` | Low-variance resampling |

---

## Experiments

### 1 — Trajectory Tracking

Both filters track the two state components over `t = 0…100` with `σ_u = σ_v = 1`. The Kalman filter produces smoother estimates; the particle filter (N = 500) follows correctly with slightly higher variance.

### 2 — Noise Sensitivity

Two extreme configurations are compared:

| Config | σ_u | σ_v | Observation |
|--------|-----|-----|-------------|
| Case 1 | 0.5 | 5.0 | Reliable dynamics, noisy sonar → filters rely on the model, smooth tracking |
| Case 2 | 5.0 | 0.5 | Erratic dynamics, precise sonar → filters rely on observations, higher MSE |

### 3 — Mean Squared Error

MSE is averaged over 50 independent simulations. The Kalman filter achieves the lowest MSE (≈ 7.8), followed by the particle filter (≈ 9.6 with N = 500). Direct inversion `B⁻¹y` yields MSE ≈ 123 — an order of magnitude worse — because it amplifies observation noise and ignores temporal dynamics.

Under **uniform noise**, the Kalman filter remains competitive (BLUE property), while the particle filter with correct uniform likelihood is sensitive to particle count due to hard-boundary weights.

### 4 — Execution Time

The Kalman filter runs in ≈ 1.9 ms (constant cost). The particle filter scales linearly with N, reaching ≈ 13 ms at N = 2000.

### 5 — Posterior Variance of z₁₀₀

The variance of `z₁₀₀ = ‖x₁₀₀‖` conditioned on `y_{1:100}` is estimated using 5000 weighted particles. The posterior is approximated via:

```
E[z]   ≈ Σ w⁽ⁱ⁾ z⁽ⁱ⁾
Var[z] ≈ Σ w⁽ⁱ⁾ (z⁽ⁱ⁾ − E[z])²
```

This demonstrates a key advantage of particle filters: estimating arbitrary nonlinear functionals of the posterior without linearization.

---
## Authors

[Noel Andolz Aguado](https://github.com/nooelanag) and [Daniel Lozano Uceda](https://github.com/dalouc)
