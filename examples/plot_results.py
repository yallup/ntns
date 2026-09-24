#!/usr/bin/env python
"""Plot interatomic-distance / energy / logZ summaries from an NTNS run.

Reads ``ns_samples.npz`` (saved by ``run_{,mala_}{dw4,lj13,lj55}.py``),
reconstructs the dead-point NSInfo, and uses BlackJAX's ``log_weights`` to:

  * histogram interatomic distances over the dead points, weighted by the mean
    importance weight at the requested inverse temperature ``--beta``
  * histogram target energies similarly
  * histogram log Z over Monte Carlo replicas of the prior-volume shrinkage

Usage:

    uv run examples/plot_results.py results/lj13_mala
    uv run examples/plot_results.py results/dw4_mala --beta 0.5
    uv run examples/plot_results.py results/lj55_mala
"""

import argparse
import os

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.special import logsumexp

from blackjax.ns.base import NSInfo, StateWithLogLikelihood
from blackjax.ns.utils import log_weights

from ntns import DoubleWell, LennardJones


def infer_target(dim: int):
    """Pick (target, n_particles, n_dim) from the flattened particle dimension."""
    if dim == 8:
        return DoubleWell(n_particles=4, n_dim=2, a=0.0, b=-4.0, c=0.9, r0=4.0), 4, 2
    if dim == 39:
        return LennardJones(n_particles=13, eps=2.0, osc_scale=0.0), 13, 3
    if dim == 165:
        return LennardJones(n_particles=55, eps=2.0, osc_scale=0.0), 55, 3
    raise ValueError(f"Unrecognised particle dimension dim={dim}")


def pairwise_distances(x_particles):
    """All pairwise distances for a single configuration (n_particles, n_dim)."""
    diff = x_particles[:, None, :] - x_particles[None, :, :]
    d = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-12)
    n = x_particles.shape[0]
    iu = jnp.triu_indices(n, k=1)
    return d[iu]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("result_dir", help="directory containing ns_samples.npz")
    p.add_argument("--n-logz-paths", type=int, default=500,
                   help="number of Monte Carlo paths for the logZ histogram")
    p.add_argument("--beta", type=float, default=1.0,
                   help="inverse temperature for the r/E weighted histograms "
                        "(1.0 = strict posterior, <1 warms it for visualisation)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    npz_path = os.path.join(args.result_dir, "ns_samples.npz")
    z = np.load(npz_path)
    positions = np.asarray(z["positions"])
    logL = np.asarray(z["logL"])
    logL_birth = np.asarray(z["logL_birth"])
    # Phase-1 prior-rejection points use blackjax's default birth value (NaN);
    # for log_weights / sample they need to read as -inf (no death contour).
    logL_birth = np.where(np.isnan(logL_birth), -np.inf, logL_birth)
    logZ_run = float(z["logZ"]) if "logZ" in z.files else float(z["logZ_int"])
    n_dead, dim = positions.shape

    target, n_particles, n_dim = infer_target(dim)
    print(f"{npz_path}: {n_dead} dead points, dim={dim} "
          f"({n_particles} particles x {n_dim}D), run logZ={logZ_run:.3f}")

    # Reconstruct NSInfo for the BlackJAX NS utilities (only the loglikelihood
    # / loglikelihood_birth fields are exercised by log_weights / sample, but
    # the NamedTuple plumbing wants the full StateWithLogLikelihood).
    particles = StateWithLogLikelihood(
        position=jnp.asarray(positions),
        logdensity=jnp.zeros(n_dead),
        loglikelihood=jnp.asarray(logL),
        loglikelihood_birth=jnp.asarray(logL_birth),
    )
    dead = NSInfo(particles=particles, update_info=None)

    rng = jax.random.PRNGKey(args.seed)
    rng, k_lw, k_lwb = jax.random.split(rng, 3)

    log_w = log_weights(k_lw, dead, shape=args.n_logz_paths)   # (n_dead, n_paths)
    logZ_paths = jnp.asarray(logsumexp(log_w, axis=0))         # (n_paths,)

    # Weighted dead-point histograms (no resampling): mean importance weight
    # per dead point at the requested beta. Avoids the discrete-resample
    # degeneracy when the posterior is sharp (e.g. LJ ground state at beta=1).
    log_w_beta = log_weights(k_lwb, dead, shape=args.n_logz_paths, beta=args.beta)
    log_w_mean = logsumexp(log_w_beta, axis=1) - jnp.log(args.n_logz_paths)
    log_w_mean = np.asarray(log_w_mean - jnp.max(log_w_mean))
    weights = np.exp(log_w_mean)
    weights = weights / weights.sum()

    configs = positions.reshape(-1, n_particles, n_dim)
    r_per_dead = np.asarray(jax.vmap(pairwise_distances)(jnp.asarray(configs)))
    n_pairs = r_per_dead.shape[1]
    r_all = r_per_dead.reshape(-1)
    r_weights = np.repeat(weights, n_pairs)
    E_all = -np.asarray(jax.vmap(target.log_prob)(jnp.asarray(positions)))
    E_weights = weights

    def _weighted_quantile_range(x, w, q_lo=1e-3, q_hi=1 - 1e-3, pad=0.05):
        """Bin range derived from weighted quantiles of x, padded by `pad` of width."""
        finite = np.isfinite(x) & np.isfinite(w) & (w > 0)
        x, w = x[finite], w[finite]
        idx = np.argsort(x)
        cw = np.cumsum(w[idx]) / w[idx].sum()
        n = len(cw)
        lo = float(x[idx][min(np.searchsorted(cw, q_lo), n - 1)])
        hi = float(x[idx][min(np.searchsorted(cw, q_hi), n - 1)])
        w_pad = (hi - lo) * pad
        return lo - w_pad, hi + w_pad

    def _weighted_fill(ax, x, weights, bins, label_x, title, color):
        x = np.asarray(x)
        w = np.asarray(weights)
        finite = np.isfinite(x) & np.isfinite(w)
        x, w = x[finite], w[finite]
        if x.size == 0 or w.sum() == 0:
            ax.text(0.5, 0.5, "no finite samples", ha="center", va="center",
                    transform=ax.transAxes)
        else:
            lo, hi = _weighted_quantile_range(x, w)
            ax.hist(x, bins=bins, range=(lo, hi), weights=w, density=True,
                    color=color, alpha=0.8)
        ax.set_xlabel(label_x)
        ax.set_ylabel("density")
        ax.set_title(title)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    _weighted_fill(axes[0], r_all, r_weights, 80, "interatomic distance $r$",
                   f"$r$ histogram, $\\beta$={args.beta}", "C0")
    _weighted_fill(axes[1], E_all, E_weights, 80, "energy $E = -\\log p(x)$",
                   f"energy histogram, $\\beta$={args.beta}", "C1")
    logZ_arr = np.asarray(logZ_paths)
    axes[2].hist(logZ_arr, bins=40, density=True, color="C2", alpha=0.8)
    axes[2].set_xlabel("$\\log Z$")
    axes[2].set_ylabel("density")
    axes[2].set_title(f"$\\log Z$ over {args.n_logz_paths} prior-volume paths")
    axes[2].axvline(logZ_run, color="k", ls="--", lw=1,
                    label=f"run logZ = {logZ_run:.3f}")
    axes[2].legend(loc="best", fontsize=8)

    fig.tight_layout()
    out_path = os.path.join(args.result_dir, "summary.png")
    fig.savefig(out_path, dpi=150)
    print(f"saved {out_path}")
    print(f"  logZ posterior over {logZ_arr.size} paths: "
          f"mean={logZ_arr.mean():.3f}  std={logZ_arr.std():.3f}")


if __name__ == "__main__":
    main()
