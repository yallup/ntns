# NTNS — Neural Transport Nested Sampling

Flow-based proposals for nested sampling. Trains a continuous normalising flow (CNF) online on live points and uses it as a proposal for constrained-prior replacement, preserving exactness via MH correction.

## Package structure

- `src/ntns/flow.py` — `ParticleFlowMatching`: EGNN velocity net trained by conditional flow matching, with zero-COM constraint, exact / Hutchinson divergence, and warm-restart support for retraining across NS iterations
- `src/ntns/egnn.py` — E(n)-equivariant graph neural network velocity field
- `src/ntns/proposals.py` — NS inner kernels: flow-IRMH with cached log_q
- `src/ntns/mala_kernel.py` — MALA-within-NS using the flow velocity as Langevin drift, with Robbins–Monro step-size adaption
- `src/ntns/targets.py` — DW4, LJ13, LJ55 targets
- `examples/run_{dw4,lj13,lj55}.py` — NTNS with the IRMH inner kernel (tsit5 ODE solver)
- `examples/run_mala_{dw4,lj13,lj55}.py` — NTNS with the MALA inner kernel (RM step-size adaption)

## Install

```bash
uv sync              # CPU
uv sync --extra cuda # GPU (CUDA 12)
```

The inner kernels rely on the `ns.from_mcmc` and `ns.irmh` modules from the `handley-lab/blackjax` fork (`irmh` branch), pinned via `[tool.uv.sources]` in `pyproject.toml`.

## Usage

```bash
# IRMH inner kernel + tsit5 solver
uv run examples/run_dw4.py
uv run examples/run_lj13.py
uv run examples/run_lj55.py

# MALA inner kernel + Robbins–Monro step-size adaption
uv run examples/run_mala_dw4.py
uv run examples/run_mala_lj13.py
uv run examples/run_mala_lj55.py
```
