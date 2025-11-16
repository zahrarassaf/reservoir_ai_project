"""
utils.py
Utility helpers for plotting and I/O.
"""
import os
import matplotlib.pyplot as plt

def ensure_dirs():
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

def save_plot(fig, fname: str):
    ensure_dirs()
    path = os.path.join("results/figures", fname)
    fig.savefig(path, bbox_inches='tight')
    print(f"Saved plot: {path}")
