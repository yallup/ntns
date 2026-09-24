"""Flow IRMH inner kernel for nested sampling."""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp


class FlowIRMHInfo(NamedTuple):
    num_inner_steps: jnp.ndarray
    total_accepted: jnp.ndarray
    acceptance_rate: jnp.ndarray


def build_flow_irmh_kernel_python(
    init_state_fn: Callable,
    flow,
    logprior_fn: Callable,
    loglikelihood_fn: Callable,
    num_inner_steps: int,
    num_delete: int,
):
    """Build a flow-based IRMH inner kernel.

    Uses the flow as an independent MH proposal. Caches log_q per chain so
    only one backward ODE is needed at init (not per inner step).
    """

    def update_function(rng_key, state, loglikelihood_0, **params):
        choice_key, key = jax.random.split(rng_key)
        particles = state.particles
        weights = (particles.loglikelihood > loglikelihood_0).astype(jnp.float32)
        weights = jnp.where(weights.sum() > 0, weights, jnp.ones_like(weights))
        start_idx = jax.random.choice(
            choice_key, len(weights), shape=(num_delete,),
            p=weights / weights.sum(), replace=True,
        )

        positions = particles.position[start_idx]
        logpriors = jax.vmap(logprior_fn)(positions)

        if flow.n_probes is not None:
            key, trace_init_key = jax.random.split(key)
            trace_keys = jax.random.split(trace_init_key, num_delete)
            log_q_current = flow.log_prob_batch(positions, trace_keys=trace_keys)
        else:
            log_q_current = flow.log_prob_batch(positions)

        total_accepted = 0
        for step in range(num_inner_steps):
            key, prop_key, u_key = jax.random.split(key, 3)

            x_prop, log_q_prop = flow.sample_and_log_prob(prop_key, num_delete)
            logprior_prop = jax.vmap(logprior_fn)(x_prop)
            loglike_prop = jax.vmap(loglikelihood_fn)(x_prop)

            log_ratio = (logprior_prop - log_q_prop) - (logpriors - log_q_current)
            log_u = jnp.log(jax.random.uniform(u_key, shape=(num_delete,)))
            accepted = (log_u < log_ratio) & (loglike_prop > loglikelihood_0)

            positions = jnp.where(accepted[:, None], x_prop, positions)
            logpriors = jnp.where(accepted, logprior_prop, logpriors)
            log_q_current = jnp.where(accepted, log_q_prop, log_q_current)

            total_accepted += int(accepted.sum())

        final_states = jax.vmap(
            lambda pos: init_state_fn(pos, loglikelihood_birth=loglikelihood_0)
        )(positions)

        info = FlowIRMHInfo(
            num_inner_steps=jnp.array(num_inner_steps),
            total_accepted=jnp.array(total_accepted),
            acceptance_rate=jnp.array(
                total_accepted / (num_inner_steps * num_delete)
            ),
        )
        return final_states, info

    return update_function
