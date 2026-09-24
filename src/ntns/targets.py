"""Particle-system targets: Lennard-Jones cluster, double-well 4-particle (DW4)."""

import jax
import jax.numpy as jnp


class LennardJones:
    """Lennard-Jones cluster with optional harmonic confining potential.

    Matches the iDEM convention (Akhound-Sadegh et al., 2024):
        E_LJ  = sum_{i<j} eps * [(rm/d_ij)^12 - 2*(rm/d_ij)^6]
        E_osc = 0.5 * osc_scale * sum_i ||x_i - x_COM||^2
        E     = E_LJ + E_osc

    Parameters
    ----------
    n_particles : int
    eps, rm : float
        LJ well depth and equilibrium distance.
    osc_scale : float
        Harmonic oscillator force constant (set 0 to disable).
    """

    def __init__(self, n_particles: int = 13, eps: float = 1.0, rm: float = 1.0, osc_scale: float = 1.0):
        self.n_particles = n_particles
        self.n_dim = 3
        self.dim = n_particles * self.n_dim
        self.eps = eps
        self.rm = rm
        self.osc_scale = osc_scale

        rows, cols = [], []
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                rows.append(i)
                cols.append(j)
        self._pair_i = jnp.array(rows)
        self._pair_j = jnp.array(cols)

    def energy(self, x):
        x = x.reshape(self.n_particles, self.n_dim)
        x = x - jnp.mean(x, axis=0, keepdims=True)
        dx = x[self._pair_i] - x[self._pair_j]
        d = jnp.sqrt(jnp.sum(dx ** 2, axis=-1))
        inv_d = self.rm / d
        e_lj = self.eps * jnp.sum(inv_d ** 12 - 2.0 * inv_d ** 6)
        e_osc = 0.5 * self.osc_scale * jnp.sum(x ** 2)
        return e_lj + e_osc

    def log_prob(self, x):
        return -self.energy(x)

    def remove_mean(self, x):
        x = x.reshape(self.n_particles, self.n_dim)
        x = x - jnp.mean(x, axis=0, keepdims=True)
        return x.reshape(-1)


class DoubleWell:
    """Double-well particle system (DW4 benchmark from iDEM / EACF).

    N particles in n_dim dimensions with pair potential
        V(r) = a*(r - r0) + b*(r - r0)^2 + c*(r - r0)^4
    summed over unique pairs i<j. Standard DW4 params (Köhler et al.):
    a=0, b=-4, c=0.9, r0=4 (N=4, 2D).
    """

    def __init__(
        self,
        n_particles: int = 4,
        n_dim: int = 2,
        a: float = 0.0,
        b: float = -4.0,
        c: float = 0.9,
        r0: float = 4.0,
    ):
        self.n_particles = n_particles
        self.n_dim = n_dim
        self.dim = n_particles * n_dim
        self.a = a
        self.b = b
        self.c = c
        self.r0 = r0

        rows, cols = [], []
        for i in range(n_particles):
            for j in range(i + 1, n_particles):
                rows.append(i)
                cols.append(j)
        self._pair_i = jnp.array(rows)
        self._pair_j = jnp.array(cols)

    def energy(self, x):
        x = x.reshape(self.n_particles, self.n_dim)
        x = x - jnp.mean(x, axis=0, keepdims=True)
        dx = x[self._pair_i] - x[self._pair_j]
        d = jnp.sqrt(jnp.sum(dx ** 2, axis=-1))
        dr = d - self.r0
        return jnp.sum(self.a * dr + self.b * dr ** 2 + self.c * dr ** 4)

    def log_prob(self, x):
        return -self.energy(x)

    def remove_mean(self, x):
        x = x.reshape(self.n_particles, self.n_dim)
        x = x - jnp.mean(x, axis=0, keepdims=True)
        return x.reshape(-1)
