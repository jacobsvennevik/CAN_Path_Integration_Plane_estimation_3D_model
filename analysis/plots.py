"""Figures for a show run. The rate maps and autocorrelograms stay in memory only here."""
import numpy as np
import matplotlib.pyplot as plt

from .gridness2d import percentile_cells


def percentile_maps(f, ac, hgs, percentiles=(10, 50, 90)):
    """One row of rate maps and one of autocorrelograms, at the given HGS percentiles."""
    idx = percentile_cells(hgs, percentiles)
    n = len(idx)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6), squeeze=False)
    for col, k in enumerate(idx):
        axes[0, col].imshow(f[..., k], origin="lower", cmap="viridis")
        axes[0, col].set_title(f"rate  p{percentiles[col]}")
        axes[1, col].imshow(ac[..., k], origin="lower", cmap="coolwarm")
        axes[1, col].set_title(f"ac  HGS={hgs[k]:.2f}")
        for ax in (axes[0, col], axes[1, col]):
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    return fig
