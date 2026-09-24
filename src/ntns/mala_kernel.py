"""MALA-within-NS using the blackjax `from_mcmc` pattern.

Mirrors `blackjax.ns.irmh` (the reference NS-from-MCMC kernel in the
custom blackjax fork). The single-chain `mala_step` is wrapped by
`blackjax.ns.from_mcmc.build_kernel`, which:

- runs `num_inner_steps` of `mala_step` per chain via `jax.lax.scan`,
- vmaps over `num_delete` chains,
- ANDs the per-step MH accept with the within-contour check via `lax.cond`,
- jit-compiles the inner kernel.

The drift comes from a flow's velocity network evaluated at a fixed `t_drift`.
The flow's parameters are threaded through the kernel's params dict each NS
iteration, so retraining between iterations does not invalidate the JIT cache
(provided the param pytree structure is preserved — i.e. warm_start training
and a fixed architecture).
"""

from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from blackjax.ns.base import (
    delete_fn as default_delete_fn,
    init_state_strategy,
)
from blackjax.ns.from_mcmc import build_kernel as from_mcmc_build_kernel
from blackjax.types import ArrayTree


__all__ = [
    "MALAState",
    "MALAStepInfo",
    "mala_init",
    "make_mala_step",
    "build_kernel",
]


class MALAState(NamedTuple):
    """Per-chain MALA state."""

    position: ArrayTree
    logdensity: float


class MALAStepInfo(NamedTuple):
    """Per-step MALA info, matching the IRMHInfo shape used by from_mcmc."""

    acceptance_rate: float
    is_accepted: bool


def mala_init(position, logdensity_fn):
    """Initialise an MALA chain state from a position.

    Mirrors `blackjax.ns.irmh.irmh_init`: the logdensity field is populated
    by calling `logdensity_fn(position)` so that the next step's MH ratio is
    consistent with the constrained-prior target.
    """
    return MALAState(position=position, logdensity=logdensity_fn(position))


def make_mala_step(net_apply: Callable, t_drift: float, project_fn: Callable | None):
    """Build a per-chain MALA step function.

    Parameters
    ----------
    net_apply
        The flow's velocity network apply function: takes
        ``({"params": flow_params}, x, t)`` and returns the velocity vector
        at ``(x, t)``. Typically ``flow.net.apply``.
    t_drift
        Time at which the velocity field is evaluated to produce the Langevin
        drift. Heuristic; tunable per problem.
    project_fn
        Optional per-sample projector applied to noise, drift and proposal
        positions. Pass ``flow.remove_mean`` for particle systems whose
        targets/priors are translation-invariant; ``None`` for ambient-R^d
        targets.

    Returns
    -------
    A callable ``mala_step(rng_key, state, logdensity_fn, *, step_size, flow_params)``
    suitable for ``blackjax.ns.from_mcmc.build_kernel`` as ``mcmc_step_fn``.
    """

    proj = project_fn if project_fn is not None else (lambda v: v)

    def mala_step(rng_key, state, logdensity_fn, *, step_size=None, flow_params=None, **_):
        # Caller (typically `update_inner_kernel_params_fn`) must supply both;
        # raise a clearer error than the cryptic JAX failure that would otherwise occur.
        if step_size is None or flow_params is None:
            raise TypeError(
                "mala_step requires keyword args step_size and flow_params; "
                "make sure update_inner_kernel_params_fn returns both keys."
            )
        # Drift via the flow's velocity field at fixed t_drift. project_fn keeps
        # both drift and noise on the constrained subspace (zero-COM for LJ).
        g_x = proj(net_apply({"params": flow_params}, state.position, t_drift))

        prop_key, u_key = jax.random.split(rng_key)
        noise = proj(jax.random.normal(prop_key, state.position.shape))
        x_prop = state.position + step_size * g_x + jnp.sqrt(2.0 * step_size) * noise
        # Idempotent re-projection: cheap and keeps the move on the manifold.
        x_prop = proj(x_prop)
        g_prop = proj(net_apply({"params": flow_params}, x_prop, t_drift))

        logdensity_prop = logdensity_fn(x_prop)

        # Closed-form Gaussian transition log-density difference.
        forward_term = -0.25 / step_size * jnp.sum(
            (x_prop - state.position - step_size * g_x) ** 2
        )
        backward_term = -0.25 / step_size * jnp.sum(
            (state.position - x_prop - step_size * g_prop) ** 2
        )
        log_alpha = (logdensity_prop - state.logdensity) + (backward_term - forward_term)
        log_alpha = jnp.where(jnp.isfinite(log_alpha), log_alpha, -jnp.inf)

        log_u = jnp.log(jax.random.uniform(u_key))
        is_accepted = log_u < log_alpha

        proposal = MALAState(position=x_prop, logdensity=logdensity_prop)
        new_state = jax.lax.cond(
            is_accepted, lambda _: proposal, lambda _: state, operand=None,
        )
        info = MALAStepInfo(
            acceptance_rate=jnp.minimum(1.0, jnp.exp(log_alpha)),
            is_accepted=is_accepted,
        )
        return new_state, info

    return mala_step


def build_kernel(
    init_state_fn: Callable,
    logdensity_fn: Callable,
    net_apply: Callable,
    t_drift: float,
    project_fn: Callable | None,
    num_inner_steps: int,
    num_delete: int,
    update_inner_kernel_params_fn: Callable,
    delete_fn: Callable = default_delete_fn,
) -> Callable:
    """Build a Nested Sampling kernel using flow-velocity-driven MALA.

    Wraps :func:`blackjax.ns.from_mcmc.build_kernel`, supplying ``mala_init``
    and a closed-over ``mala_step`` as the MCMC primitives. The returned
    kernel has the standard adaptive-NS signature
    ``(rng_key, state) -> (new_state, NSInfo)``.

    Parameters
    ----------
    init_state_fn
        Per-particle NS state initialiser, typically a partial of
        ``init_state_strategy`` with logprior_fn and loglikelihood_fn bound.
    logdensity_fn
        Log-density of the unconstrained target; for nested sampling this is
        the log-prior. The constraint is layered on by ``from_mcmc`` via the
        within-contour check.
    net_apply
        Flow velocity network apply function (see ``make_mala_step``).
    t_drift
        Time at which the velocity field is evaluated.
    project_fn
        Optional per-sample projector (e.g. ``flow.remove_mean`` for LJ).
    num_inner_steps
        MALA steps per chain per NS iteration.
    num_delete
        Particles replaced per NS iteration.
    update_inner_kernel_params_fn
        Hook called between NS iterations; must return a dict containing at
        least ``step_size`` and ``flow_params`` so the next call to
        ``mala_step`` sees the latest values.
    delete_fn
        Particle-deletion strategy. Defaults to BlackJAX's ``delete_fn``.
    """
    mala_step = make_mala_step(net_apply, t_drift, project_fn)
    return from_mcmc_build_kernel(
        init_state_fn=init_state_fn,
        logdensity_fn=logdensity_fn,
        mcmc_init_fn=mala_init,
        mcmc_step_fn=mala_step,
        num_inner_steps=num_inner_steps,
        update_inner_kernel_params_fn=update_inner_kernel_params_fn,
        num_delete=num_delete,
        delete_fn=delete_fn,
    )
