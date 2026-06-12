# CAN_Path_Integration_Plane_estimation_3D_model

A continuous-attractor model of 3D grid cells where path integration is done relative to a
Bayesian-estimated reference plane instead of a single global lattice. The goal is to explain
why grid fields in 3D are locally ordered but globally unordered : if the brain integrates relative to a noisily estimated plane, local spacing survives
but global order breaks down.

Two parts: (1) a 3D CAN on the torus **T³** built with the **MADE**(Claudi et al., 2025). The plan is to change it a little bit with Burak & Fiete (2009) inhibitory centre-surrond inhibition kerne; and (2) a recursive
Bingham filter on S² (Kurz et al., 2014) that estimates the reference-plane normal from motion
and keeps a running belief. This replaces the fixed projection + injected von Mises–Fisher noise of
Gong & Yu (2021) with a genuinely 3D integrator and a real estimator.

> **Setup note:** the MADE package is patched locally at
> `./.venv/lib/python3.14/site-packages/made/manifolds.py` (Python 3.14 venv).
The MADE package used is a bit changed, so not the one that you get from poå


## Files

**Manifold & network**
- `torus3D_manifold.py` — defines the T³ torus and neuron coordinates.
- `metric3D.py` — twisted-torus distance the connectivity is built from (hexagonal in θ₁θ₂; θ₃ not yet coupled).
- `metrics.py` — small angular-difference helper for the error metric.
- `CAN3D.py` — one continuous attractor network (single bump).
- `QAN3D.py` — six CANs (3 axes × ± direction) giving velocity-driven movement.
- `torch_backend.py` — runs the network on GPU/CPU using an FFT recurrence; decodes the bump position.

**Plane filter & integration**
- `plane_estimation.py` — the recursive Bingham filter on S² (estimates the plane normal n̂).
- `path_integration.py` — glues it together: filter → n̂ → rotate velocity → drive the CAN → decode.

**Experiments**
- `base.py` — shared run loop and the `.npz` output format the scorer reads.
- `config.py` — single source of network/experiment parameters.
- `arena_2d.py` — the flat-floor 2D baseline (a reflecting random walk).

**Scoring**
- `gongyu_scoring.py` — Gong & Yu's structure-score code (FCC/HCP/columnar), ported with fixes.
- `scoring.py` — the project's scoring API: rate maps → autocorrelation → structure scores.
- `prototypes.py` — ideal FCC/HCP/COL/random lattices to compare real runs against.

**Other**
- `visualize3D.py`, `utils3D.py` — plotting and batch-simulation helpers.
- `notebooks/` — step-by-step checks (connectivity, filter, manifold, full pipeline, scoring).
- Outputs go in `results/` (gitignored). See `docs/implementation/` for the full per-module reference.

## Key references

- Burak & Fiete (2009), *PLoS Comput. Biol.* — the base grid-cell CAN.
- Claudi, Chandra & Fiete (2025), *eLife* (reviewed preprint) — the MADE framework. https://doi.org/10.7554/eLife.107224.1
- Gong & Yu (2021), *Front. Comput. Neurosci.* — the plane-based 3D model we refine.
- Kurz, Gilitschenski, Julier & Hanebeck (2014), *J. Adv. Inf. Fusion* — the recursive Bingham filter.
- Ginosar et al. (2021), *Nature*; Grieves et al. (2021), *Nat. Neurosci.* — the 3D grid-cell findings being explained.

## Acknowledgements

- CAN model started from changmin-yu's Burak–Fiete implementation:
  https://github.com/changmin-yu/grid-cell-models-python/blob/main/burak_fiete_2009.py
- Scoring code ported from Gong & Yu: https://github.com/gongziyida/GridCells3D
- Plane filter follows Kurz et al. (2014). *(The project began as a fork of the Fernández-León et al.
  2022 controller; its place-cell anchoring has since been replaced by the Bingham plane filter.)*