#!/usr/bin/env python
"""NTNS-IRMH on DW4 (4 particles, 2D, 8D total).

NTNS pattern: prior rejection -> EGNN flow trained on live points -> flow used
as an independent Metropolis-Hastings proposal (tsit5 ODE solver, exact log-det
trace).
"""

import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import tqdm

from blackjax.ns.adaptive import (
    build_kernel as build_adaptive_kernel,
    init as adaptive_init,
)
from blackjax.ns.base import (
    NSInfo,
    init_state_strategy,
    delete_fn as default_delete_fn,
)
from blackjax.ns.integrator import NSIntegrator
from blackjax.ns.utils import finalise
from jax.scipy.special import logsumexp

from ntns import DoubleWell, ParticleFlowMatching
from ntns.proposals import build_flow_irmh_kernel_python, FlowIRMHInfo

# Config -----------------------------------------------------------------
N_PARTICLES, N_DIM = 4, 2
DIM = N_PARTICLES * N_DIM  # 8

NLIVE, NDELETE, NUM_INNER_STEPS = 2048, 1024, 20
SEED = 42

FLOW_EPOCHS, FLOW_BATCH, FLOW_LR, FLOW_STEPS = 300, 512, 1e-3, 25
EGNN_LAYERS, EGNN_HIDDEN = 5, 128

# Unit-Gaussian prior; +0.5||x||² in loglikelihood cancels prior so NS samples
# the pure DW4 Boltzmann exp(-E_DW(x)).
PRIOR_SIGMA = 1.0
PRIOR_FACTOR = 100  # phase-1 oversampling

OUTDIR = "results/dw4"
os.makedirs(OUTDIR, exist_ok=True)

# Target + prior ---------------------------------------------------------
target = DoubleWell(n_particles=N_PARTICLES, n_dim=N_DIM, a=0.0, b=-4.0, c=0.9, r0=4.0)


def loglikelihood_fn(x):
    return target.log_prob(x) + 0.5 * jnp.sum(x ** 2)


LOG_PRIOR_NORM = -0.5 * DIM * jnp.log(2 * jnp.pi) - DIM * jnp.log(PRIOR_SIGMA)


def logprior_fn(x):
    return LOG_PRIOR_NORM - 0.5 * jnp.sum((x / PRIOR_SIGMA) ** 2)


def sample_prior(key, n):
    x = jax.random.normal(key, (n, DIM)) * PRIOR_SIGMA
    return jax.vmap(target.remove_mean)(x)


# NS init ----------------------------------------------------------------
init_state_fn = partial(init_state_strategy, logprior_fn=logprior_fn, loglikelihood_fn=loglikelihood_fn)
rng_key = jax.random.PRNGKey(SEED)
delete_fn = partial(default_delete_fn, num_delete=NDELETE)

# Phase 1: prior rejection -----------------------------------------------
t0 = time.time()
rng_key, prior_key = jax.random.split(rng_key)
n_prior = PRIOR_FACTOR * NLIVE
prior_samples = sample_prior(prior_key, n_prior)
prior_states = jax.vmap(init_state_fn)(prior_samples)

_, top_idx = jax.lax.top_k(prior_states.loglikelihood, NLIVE)
positions = prior_states.position[top_idx]

mask = jnp.ones(n_prior, dtype=bool).at[top_idx].set(False)
dead_idx = jnp.where(mask, size=n_prior - NLIVE, fill_value=0)[0]
dead_states = jax.tree.map(lambda x: x[dead_idx], prior_states)
dead_sort = jnp.argsort(dead_states.loglikelihood)
dead_states = jax.tree.map(lambda x: x[dead_sort], dead_states)

dead_list = [NSInfo(dead_states, FlowIRMHInfo(
    num_inner_steps=jnp.array(0),
    total_accepted=jnp.array(NLIVE),
    acceptance_rate=jnp.array(NLIVE / n_prior),
))]

state = adaptive_init(positions, init_state_fn=jax.vmap(init_state_fn))

# Advance integrator to reflect phase-1 contraction (n_prior -> NLIVE)
num_live_p1 = jnp.arange(n_prior, NLIVE, -1)
delta_logX_p1 = -1.0 / num_live_p1
logX_p1 = jnp.cumsum(delta_logX_p1)
log_delta_X_p1 = logX_p1 + jnp.log1p(-jnp.exp(delta_logX_p1))
logZ_p1 = logsumexp(dead_states.loglikelihood + log_delta_X_p1)
logX_final_p1 = logX_p1[-1]
logZ_live_p1 = (
    logsumexp(state.particles.loglikelihood)
    - jnp.log(state.particles.loglikelihood.shape[0])
    + logX_final_p1
)
state = state._replace(integrator=NSIntegrator(logX_final_p1, logZ_p1, logZ_live_p1))

# Phase 2: flow IRMH -----------------------------------------------------
flow = ParticleFlowMatching(
    n_particles=N_PARTICLES, n_dim=N_DIM, n_steps=FLOW_STEPS, seed=SEED,
    hidden_nf=EGNN_HIDDEN, n_layers=EGNN_LAYERS,
)
flow.train(positions, n_epochs=FLOW_EPOCHS, batch_size=FLOW_BATCH, lr=FLOW_LR)


def flow_update_fn(rng_key, state, info, inner_kernel_params=None):
    flow.train(
        state.particles.position,
        n_epochs=FLOW_EPOCHS, batch_size=FLOW_BATCH, lr=FLOW_LR,
        warm_start=False,
    )
    return {}


inner_kernel = build_flow_irmh_kernel_python(
    init_state_fn=init_state_fn, flow=flow,
    logprior_fn=logprior_fn, loglikelihood_fn=loglikelihood_fn,
    num_inner_steps=NUM_INNER_STEPS, num_delete=NDELETE,
)
flow_kernel = build_adaptive_kernel(delete_fn, inner_kernel, update_inner_kernel_params_fn=flow_update_fn)

acc_rates = []
with tqdm.tqdm(desc="NTNS-IRMH", unit=" pts") as pbar:
    while not (state.integrator.logZ_live - state.integrator.logZ < -3):
        rng_key, subkey = jax.random.split(rng_key)
        state, info = flow_kernel(subkey, state)
        dead_list.append(info)
        acc = float(info.update_info.acceptance_rate)
        acc_rates.append(acc)
        pbar.update(NDELETE)
        pbar.set_postfix(logZ=f"{float(state.integrator.logZ):.2f}", acc=f"{acc:.3f}")
        if acc == 0.0:
            break

ns_run = finalise(state, dead_list, update_info=False)

np.savez(
    f"{OUTDIR}/ns_samples.npz",
    positions=np.asarray(ns_run.particles.position),
    logL=np.asarray(ns_run.particles.loglikelihood),
    logL_birth=np.asarray(ns_run.particles.loglikelihood_birth),
    logZ=float(state.integrator.logZ),
    acc_rates=np.array(acc_rates),
)
print(f"\nlogZ = {float(state.integrator.logZ):.3f}  wall = {time.time() - t0:.1f}s")
