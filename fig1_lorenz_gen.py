"""
FIG01_Lorenz – Python re‑implementation of Brunton et al. (2016) MATLAB demo.
This script reproduces the data generation and key visualizations for the
HAVOK analysis of the Lorenz system (Fig. 01 in the paper).

Run with:
    python fig01_lorenz.py

Author: ChatGPT (2025‑05‑31)
Dependencies: numpy, scipy (>=1.4), matplotlib.
"""

from __future__ import annotations
import os
from typing import Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import svd
from scipy.signal import StateSpace, lsim

# =====================  HELPERS  ==========================================


def lorenz(t: float, state: np.ndarray, sigma: float = 10.0, beta: float = 8.0 / 3.0, rho: float = 28.0) -> list[float]:
    """Lorenz‐63 chaotic dynamics."""
    x, y, z = state
    return [sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z]


def optimal_svht_coef(beta: float) -> float:
    """Approximate Gavish–Donoho optimal hard‑threshold coefficient.

    This cubic fit is accurate in the range 0 < β ≤ 1 (see sup. Fig. 3)."""
    return 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43


def build_hankel(series: np.ndarray, stack_max: int) -> np.ndarray:
    """Return Hankel/trajectory matrix used for time‑delay embedding."""
    N = len(series)
    return np.stack([series[k:N - stack_max + k] for k in range(stack_max)])


def central_finite_diff(V: np.ndarray, dt: float) -> np.ndarray:
    """Fourth‑order central finite difference (matches MATLAB code)."""
    dV = np.zeros_like(V)
    dV[2:-2] = (-V[4:] + 8 * V[3:-1] - 8 * V[1:-3] + V[:-4]) / (12 * dt)
    # Fallback to lower order near edges
    dV[0] = (V[1] - V[0]) / dt
    dV[1] = (V[2] - V[0]) / (2 * dt)
    dV[-2] = (V[-1] - V[-3]) / (2 * dt)
    dV[-1] = (V[-1] - V[-2]) / dt
    return dV


def pool_data(X: np.ndarray, polyorder: int = 1) -> np.ndarray:
    """Build a small library – here only constant and linear terms."""
    n_samples, _ = X.shape
    Theta_parts = [np.ones((n_samples, 1)), X]
    # Extend with quadratic/cubic terms if polyorder > 1
    return np.hstack(Theta_parts)


def sparsify_dynamics(Theta: np.ndarray, dx: np.ndarray, lam: float, n_iter: int = 10) -> np.ndarray:
    """Sequentially‑thresholded least‑squares (STLS) regression.

    Supports both 1‑D (n_samples,) and 2‑D (n_samples, n_targets) *dx*."""
    if dx.ndim == 1:
        dx = dx[:, None]  # cast to 2‑D for uniform treatment

    # Initial least‑squares solution – one column per target
    Xi = np.linalg.lstsq(Theta, dx, rcond=None)[0]  # shape (m, n_targets)

    for _ in range(n_iter):
        small = np.abs(Xi) < lam
        Xi[small] = 0.0
        for j in range(Xi.shape[1]):
            big = ~small[:, j]
            if not np.any(big):
                continue
            Xi[big, j] = np.linalg.lstsq(Theta[:, big], dx[:, j], rcond=None)[0]
    return Xi.squeeze()  # drop singletons when returning

# =====================  MAIN PIPELINE  ====================================


def make_figures(fig_dir: str, tspan: np.ndarray, xdat: np.ndarray, V: np.ndarray,
                  x: np.ndarray, y: np.ndarray, U: np.ndarray, r: int) -> None:
    """Generate and save figure panels 1–7 (mirrors MATLAB layout)."""
    os.makedirs(fig_dir, exist_ok=True)

    # Part 1 – Lorenz attractor -------------------------------------------------
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*xdat[:200_000].T, color="k", lw=1.5)
    ax.view_init(elev=12, azim=-5)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    fig.savefig(os.path.join(fig_dir, "lorenz_attractor.png"), dpi=300)
    plt.close(fig)

    # Part 2 – Time series ------------------------------------------------------
    plt.figure(figsize=(5.5, 3))
    plt.plot(tspan, xdat[:, 0], "k", lw=2)
    plt.xlabel("t"); plt.ylabel("x")
    plt.xlim(0, 100); plt.ylim(-20, 20)
    plt.yticks([-20, -10, 0, 10, 20])
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "lorenz_timeseries.png"), dpi=300)
    plt.close()

    # Part 3 – Embedded attractor ----------------------------------------------
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*V[:170_000, :3].T, color="k", lw=1.5)
    ax.view_init(elev=22, azim=34)
    ax.set_xlabel("v₁"); ax.set_ylabel("v₂"); ax.set_zlabel("v₃")
    fig.savefig(os.path.join(fig_dir, "embedded_attractor.png"), dpi=300)
    plt.close(fig)

    # Part 4 – Model vs. data ---------------------------------------------------
    L1 = np.arange(300, 25_000)
    L2 = np.arange(300, 25_000, 50)
    plt.figure(figsize=(5.5, 5))
    plt.subplot(2, 1, 1)
    plt.plot(tspan[L1], x[L1, 0], color=[0.4, 0.4, 0.4], lw=2.5)
    plt.plot(tspan[L2], y[L2, 0], ".", color=[0, 0, 0.5], ms=5)
    plt.ylabel("v₁"); plt.xlim(0, tspan[L1][-1]); plt.ylim(-0.0051, 0.005)
    plt.subplot(2, 1, 2)
    plt.plot(tspan[L1], x[L1, -1], color=[0.5, 0, 0], lw=1.5)
    plt.xlabel("t"); plt.ylabel(f"v_{r}")
    plt.xlim(0, tspan[L1][-1]); plt.ylim(-0.025, 0.024)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "model_vs_data.png"), dpi=300)
    plt.close()

    # Part 5 – Reconstructed attractor -----------------------------------------
    L3 = np.arange(300, 50_000)
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*y[L3, :3].T, color=[0, 0, 0.5], lw=1.5)
    ax.set_xlabel("v₁"); ax.set_ylabel("v₂"); ax.set_zlabel("v₃")
    ax.view_init(elev=22, azim=34)
    fig.savefig(os.path.join(fig_dir, "reconstructed_attractor.png"), dpi=300)
    plt.close(fig)

    # Part 6 – Forcing statistics ----------------------------------------------
    plt.figure(figsize=(5.5, 3))
    Vtest = np.std(V[:, -1]) * np.random.randn(200_000)
    h, bins = np.histogram(V[:, -1] - V[:, -1].mean(), bins=np.arange(-0.03, 0.0301, 0.0025))
    h_norm, bins2 = np.histogram(Vtest - Vtest.mean(), bins=np.arange(-0.02, 0.0201, 0.0025))
    bc = (bins[:-1] + bins[1:]) / 2
    bc2 = (bins2[:-1] + bins2[1:]) / 2
    plt.semilogy(bc2, h_norm / h_norm.sum(), "--", color=[0.2, 0.2, 0.2], lw=1.5, label="Normal dist.")
    plt.semilogy(bc, h / h.sum(), color=[0.5, 0, 0], lw=1.5, label="Lorenz forcing")
    plt.xlabel(f"v_{r}"); plt.ylabel("Probability")
    plt.xlim(-0.025, 0.025); plt.ylim(1e-4, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "forcing_stats.png"), dpi=300)
    plt.close()

    # Part 7 – U modes ---------------------------------------------------------
    plt.figure(figsize=(5.5, 3))
    CC = np.array([[2, 15, 32], [2, 35, 92], [22, 62, 149],
                   [41, 85, 180], [83, 124, 213], [112, 148, 223],
                   [114, 155, 215]]) / 255.0
    for k in range(5):
        plt.plot(U[:, k], lw=1.5 + 2 * (k + 1) / 30, color=CC[k])
    if U.shape[1] >= 6:
        plt.plot(U[:, 5], color=[0.5, 0.5, 0.5], lw=1.5)
    if U.shape[1] >= 15:
        plt.plot(U[:, 14], color=[0.75, 0, 0], lw=1.5)
    plt.xlabel("time index, k"); plt.ylabel("u_r")
    plt.legend(["r=1", "r=2", "r=3", "r=4", "r=5", "…", "r=15"], loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "U_modes.png"), dpi=300)
    plt.close()


# ----------------------------------------------------------------------------


def main() -> None:
    # 1. Lorenz integration ----------------------------------------------------
    sigma, beta, rho = 10.0, 8.0 / 3.0, 28.0
    x0 = np.array([-8.0, 8.0, 27.0])
    dt = 1e-3
    tspan = np.arange(dt, 200.0 + dt, dt)

    sol = solve_ivp(lambda t, x: lorenz(t, x, sigma, beta, rho),
                    (tspan[0], tspan[-1]), x0,
                    t_eval=tspan, rtol=1e-12, atol=1e-12 * np.ones(3))
    xdat: np.ndarray = sol.y.T  # shape (N, 3)

    # 2. Time‑delay embedding ---------------------------------------------------
    stack_max = 100
    H = build_hankel(xdat[:, 0], stack_max)
    U, S, Vh = svd(H, full_matrices=False)
    beta_ratio = H.shape[0] / H.shape[1]
    thresh = optimal_svht_coef(beta_ratio) * np.median(S)
    r = min(int((S > thresh).sum()), 15)  # r ≤ 15 to match MATLAB example

    V = Vh.T[:, :r]              # rows = time, cols = delay coordinates
    dV = central_finite_diff(V, dt)
    x = V[2:-2]                  # trim to align with derivative indices
    dx = dV[2:-2]

    # 3. HAVOK linear model -----------------------------------------------------
    Theta = pool_data(x, polyorder=1)
    norm_theta = np.linalg.norm(Theta, axis=0)
    Theta_n = Theta / norm_theta

    Xi = np.zeros((Theta.shape[1], r - 1))
    for k in range(r - 1):
        Xi[:, k] = sparsify_dynamics(Theta_n, dx[:, k], lam=0.0)
    Xi /= norm_theta[:, None]  # denormalize

    # Build state-space matrices following MATLAB ordering
    # Xi rows: 0 = constant term, 1..r = delay coordinates
    Atemp = Xi[1:r + 1, :r - 1].T        # shape (r-1) × r
    B = Atemp[:, -1].reshape(-1, 1)      # last column is forcing term (v_r)
    A = Atemp[:, :-1]                    # remaining columns form A

    sys = StateSpace(A, B, np.eye(r - 1), np.zeros((r - 1, 1)))
    (A, B, np.eye(r - 1), np.zeros((r - 1, 1)))

    # Simulate forced linear system -------------------------------------------
    L = np.arange(50_000)
    t_sim = dt * L
    _, y, _ = lsim(sys, U=V[L, -1], T=t_sim, X0=V[0, :r - 1])

    # 4. Figures ---------------------------------------------------------------
    make_figures("./figures", tspan, xdat, V, x, y, U, r)

    print("Finished. Figures saved in ./figures/")


if __name__ == "__main__":
    main()
