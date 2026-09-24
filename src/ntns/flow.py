"""Continuous flow matching for particle systems (zero-COM, EGNN velocity field)."""

import jax
import jax.numpy as jnp
import distrax
import diffrax
import optax

from ntns.egnn import EGNNVelocity


def _exact_trace(vel_fn, x, t):
    """Trace(dv/dx) via exact VJPs (d calls)."""
    _, vjp_fn = jax.vjp(lambda x_: vel_fn(x_, t), x)
    eye = jnp.eye(x.shape[0])
    rows = jax.vmap(lambda e: vjp_fn(e)[0])(eye)
    return jnp.trace(rows)


def _hutchinson_trace(vel_fn, x, t, key, n_probes=1):
    """Hutchinson estimator of Trace(dv/dx) using Rademacher probes."""
    d = x.shape[0]
    vs = 2 * jax.random.bernoulli(key, shape=(n_probes, d)).astype(x.dtype) - 1
    _, vjp_fn = jax.vjp(lambda x_: vel_fn(x_, t), x)
    return jnp.mean(jax.vmap(lambda v: jnp.dot(v, vjp_fn(v)[0]))(vs))


class ParticleFlowMatching:
    """CFM with zero-COM constraint and EGNN velocity field.

    Trains a velocity field by conditional flow matching (straight-line
    interpolant from a unit Gaussian prior to data), then provides
    `sample_and_log_prob` (forward ODE) and `log_prob_batch` (backward ODE)
    for use as an MH proposal. Set `n_probes` to enable Hutchinson trace
    estimation in place of exact VJP-based traces (cheaper at high d).
    """

    def __init__(
        self,
        n_particles: int,
        n_dim: int = 3,
        n_steps: int = 100,
        seed: int = 0,
        n_probes: int | None = None,
        solver: str = "tsit5",
        rtol: float = 1e-5,
        atol: float = 1e-5,
        max_steps: int = 4096,
        **egnn_kwargs,
    ):
        self.n_particles = n_particles
        self.n_dim = n_dim
        self.dim = n_particles * n_dim
        self.n_steps = n_steps
        self.n_probes = n_probes
        self.rng = jax.random.PRNGKey(seed)
        self.solver_name = solver.lower()
        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps

        self.net = EGNNVelocity(n_particles=n_particles, n_dimension=n_dim, **egnn_kwargs)
        self.params = None
        self.opt_state = None
        self._tx = None

    # -- helpers ---------------------------------------------------------

    def _make_solver(self):
        return {"tsit5": diffrax.Tsit5, "dopri5": diffrax.Dopri5, "euler": diffrax.Euler}[self.solver_name]()

    def _is_adaptive(self):
        return self.solver_name in ("tsit5", "dopri5")

    def _init_params(self, key):
        return self.net.init(key, jnp.zeros(self.dim), jnp.array(0.0))["params"]

    def _apply(self, params, x, t):
        return self.net.apply({"params": params}, x, t)

    def remove_mean(self, x):
        """Project a flat (dim,) vector to zero center-of-mass."""
        pos = x.reshape(self.n_particles, self.n_dim)
        pos = pos - jnp.mean(pos, axis=0, keepdims=True)
        return pos.reshape(-1)

    # -- ODE solves ------------------------------------------------------

    def _solve(self, params, y0_x, t0, t1, dt0_sign, trace_key=None):
        """Integrate (x, log_det) from t0 to t1. Returns (x_final, log_det)."""
        apply_fn = lambda x_, t_: self._apply(params, x_, t_)
        use_hutchinson = self.n_probes is not None and trace_key is not None
        if use_hutchinson:
            eps = 2 * jax.random.bernoulli(trace_key, shape=(self.n_probes, self.dim)).astype(y0_x.dtype) - 1

            def ode_fn(t, y, args):
                x, _ = y
                v, vjp_fn = jax.vjp(lambda x_: apply_fn(x_, t), x)
                trace = jnp.mean(jnp.sum(jax.vmap(lambda e: vjp_fn(e)[0])(eps) * eps, axis=-1))
                return (v, trace)
        else:
            def ode_fn(t, y, args):
                x, _ = y
                return (apply_fn(x, t), _exact_trace(apply_fn, x, t))

        term = diffrax.ODETerm(ode_fn)
        solver = self._make_solver()
        y0 = (y0_x, jnp.array(0.0))
        if self._is_adaptive():
            sol = diffrax.diffeqsolve(
                term, solver, t0=t0, t1=t1, dt0=dt0_sign * 0.1, y0=y0,
                stepsize_controller=diffrax.PIDController(rtol=self.rtol, atol=self.atol),
                max_steps=self.max_steps,
                saveat=diffrax.SaveAt(t1=True),
            )
        else:
            sol = diffrax.diffeqsolve(
                term, solver, t0=t0, t1=t1, dt0=dt0_sign / self.n_steps, y0=y0,
                saveat=diffrax.SaveAt(t1=True),
            )
        return sol.ys[0][-1], sol.ys[1][-1]

    def _solve_ode(self, params, z, trace_key=None):
        x, log_det = self._solve(params, z, 0.0, 1.0, +1.0, trace_key)
        return self.remove_mean(x), log_det

    def _solve_ode_backward(self, params, x, trace_key=None):
        return self._solve(params, x, 1.0, 0.0, -1.0, trace_key)

    # -- public API ------------------------------------------------------

    def log_prob(self, x, trace_key=None):
        prior = distrax.MultivariateNormalDiag(loc=jnp.zeros(self.dim), scale_diag=jnp.ones(self.dim))
        z, log_det_back = self._solve_ode_backward(self.params, x, trace_key=trace_key)
        return prior.log_prob(z) + log_det_back

    def log_prob_batch(self, xs, trace_keys=None):
        if trace_keys is not None:
            return jax.vmap(lambda x, k: self.log_prob(x, trace_key=k))(xs, trace_keys)
        return jax.vmap(self.log_prob)(xs)

    def sample_and_log_prob(self, key, n):
        if self.n_probes is not None:
            sample_key, trace_key = jax.random.split(key)
            z = jax.random.normal(sample_key, (n, self.dim))
            trace_keys = jax.random.split(trace_key, n)
        else:
            z = jax.random.normal(key, (n, self.dim))
            trace_keys = None

        z = jax.vmap(self.remove_mean)(z)
        prior = distrax.MultivariateNormalDiag(loc=jnp.zeros(self.dim), scale_diag=jnp.ones(self.dim))
        log_prior_z = prior.log_prob(z)

        if trace_keys is not None:
            x, log_det = jax.vmap(lambda zi, ki: self._solve_ode(self.params, zi, trace_key=ki))(z, trace_keys)
        else:
            x, log_det = jax.vmap(lambda zi: self._solve_ode(self.params, zi))(z)
        return x, log_prior_z - log_det

    # -- training --------------------------------------------------------

    def train(
        self,
        data,
        n_epochs: int = 100,
        batch_size: int = 64,
        lr: float = 1e-2,
        noise: float = 1e-3,
        warm_start: bool = False,
    ):
        data = jax.vmap(self.remove_mean)(data)

        self.rng, init_key = jax.random.split(self.rng)
        if self.params is None or not warm_start:
            self.params = self._init_params(init_key)

        n_data = data.shape[0]
        batch_size = min(batch_size, n_data)
        total_steps = n_epochs * max(1, n_data // batch_size)
        schedule = optax.cosine_decay_schedule(init_value=lr, decay_steps=total_steps)
        self._tx = optax.apply_if_finite(
            optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(schedule)),
            max_consecutive_errors=100,
        )
        self.opt_state = self._tx.init(self.params)

        remove_mean = self.remove_mean

        @jax.jit
        def step_fn(params, opt_state, x1, key):
            def loss_fn(params):
                k1, k2, k3 = jax.random.split(key, 3)
                tau = jax.random.uniform(k1, (x1.shape[0], 1))
                x0 = jax.vmap(remove_mean)(jax.random.normal(k2, x1.shape))
                eps = jax.vmap(remove_mean)(jax.random.normal(k3, x1.shape))
                xt = tau * x1 + (1 - tau) * x0 + noise * eps
                v_pred = jax.vmap(lambda xi, ti: self._apply(params, xi, ti))(xt, tau.squeeze())
                return jnp.mean((v_pred - (x1 - x0)) ** 2)

            loss, grads = jax.value_and_grad(loss_fn)(params)
            updates, opt_state_new = self._tx.update(grads, opt_state, params)
            return optax.apply_updates(params, updates), opt_state_new, loss

        losses = []
        for _ in range(n_epochs):
            self.rng, perm_key = jax.random.split(self.rng)
            perm = jax.random.permutation(perm_key, n_data)
            for i in range(0, n_data - batch_size + 1, batch_size):
                self.rng, sk = jax.random.split(self.rng)
                self.params, self.opt_state, loss = step_fn(
                    self.params, self.opt_state, data[perm[i:i + batch_size]], sk,
                )
                losses.append(float(loss))
        return losses
