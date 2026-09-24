"""E(n) Equivariant Graph Neural Network velocity field for flow matching.

Minimal JAX/Flax port of the EGNN architecture from Satorras et al. (2021),
following the iDEM implementation (Akhound-Sadegh et al., 2024).

For use as velocity network in ContinuousFlowMatching on particle systems
(e.g. Lennard-Jones clusters) where E(3) equivariance is required.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn


class EGCLLayer(nn.Module):
    """Equivariant Graph Convolutional Layer.

    Single message-passing step that updates node features h (invariant)
    and coordinates x (equivariant) on a fully-connected graph.

    Following iDEM: coord_diff is damped-normalized by 1/(||r||+1),
    and the coord MLP last layer uses small-scale init.
    """

    hidden_nf: int = 128
    recurrent: bool = True
    tanh: bool = False
    coords_range: float = 1.0

    @nn.compact
    def __call__(self, h, x, senders, receivers):
        """
        h: (N, hidden_nf) node features
        x: (N, 3) positions
        senders, receivers: (n_edges,) index arrays

        Returns: (h', x')
        """
        n_nodes = h.shape[0]

        # Pairwise geometry from current coords
        coord_diff = x[senders] - x[receivers]  # (n_edges, 3)
        radial = jnp.sum(coord_diff**2, axis=-1, keepdims=True)  # (n_edges, 1)
        norm = jnp.sqrt(radial + 1e-8)
        coord_diff_norm = coord_diff / (norm + 1)  # damped normalization

        # --- Edge model: m_ij = phi_e(h_i, h_j, ||r_ij||^2) ---
        edge_input = jnp.concatenate(
            [h[senders], h[receivers], radial], axis=-1
        )
        m_ij = nn.silu(nn.Dense(self.hidden_nf)(edge_input))
        m_ij = nn.silu(nn.Dense(self.hidden_nf)(m_ij))

        # --- Coord model: x_i += sum_j coord_diff_norm * phi_x(m_ij) ---
        trans = nn.silu(nn.Dense(self.hidden_nf)(m_ij))
        trans = nn.Dense(
            1,
            use_bias=False,
            kernel_init=nn.initializers.variance_scaling(
                0.001, "fan_in", "uniform"
            ),
        )(trans)  # (n_edges, 1)

        if self.tanh:
            trans = jnp.tanh(trans) * self.coords_range

        x_update = coord_diff_norm * trans  # (n_edges, 3)
        agg_x = jnp.zeros_like(x).at[senders].add(x_update)
        x = x + agg_x

        # --- Node model: h_i = h_i + phi_h(h_i, sum_j m_ij) ---
        agg_m = jnp.zeros((n_nodes, self.hidden_nf)).at[senders].add(m_ij)
        node_input = jnp.concatenate([h, agg_m], axis=-1)
        h_new = nn.silu(nn.Dense(self.hidden_nf)(node_input))
        h_new = nn.Dense(self.hidden_nf)(h_new)

        if self.recurrent:
            h_new = h + h_new

        return h_new, x


class EGNNVelocity(nn.Module):
    """EGNN velocity field for flow matching on particle systems.

    E(n) equivariant: v(Rx + b, t) = Rv(x, t).
    Outputs zero-mean velocities (translation invariant).

    Parameters
    ----------
    n_particles : int
        Number of particles (e.g. 13, 38, 55 for LJ clusters).
    n_dimension : int
        Spatial dimension (default 3).
    hidden_nf : int
        Hidden feature dimension.
    n_layers : int
        Number of EGCL message-passing layers.
    recurrent : bool
        Skip connections on node features.
    tanh : bool
        Apply tanh to coord updates for stability.
    coords_range : float
        Scale for tanh coord clamping (divided by n_layers).
    """

    n_particles: int
    n_dimension: int = 3
    hidden_nf: int = 128
    n_layers: int = 5
    recurrent: bool = True
    tanh: bool = False
    coords_range: float = 15.0
    cond_dim: int = 0

    def setup(self):
        rows, cols = [], []
        for i in range(self.n_particles):
            for j in range(self.n_particles):
                if i != j:
                    rows.append(i)
                    cols.append(j)
        self._senders = jnp.array(rows)
        self._receivers = jnp.array(cols)

    @nn.compact
    def __call__(self, x, t, cond=None):
        """
        x : (N*3,) or (N, 3)  particle positions
        t : scalar             flow time
        cond : (cond_dim,) or scalar, optional conditioning vector (e.g. logLstar)

        Returns : same shape as x, velocity vectors
        """
        flat_input = x.ndim == 1
        if flat_input:
            x = x.reshape(self.n_particles, self.n_dimension)

        N = self.n_particles

        # Node features: time (+ cond) broadcast to all nodes
        t = jnp.atleast_1d(t).squeeze()
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("cond_dim>0 requires cond input")
            c = jnp.atleast_1d(cond).reshape(-1)  # (cond_dim,)
            tc = jnp.concatenate([jnp.array([t]), c])  # (1 + cond_dim,)
            h = jnp.broadcast_to(tc, (N, tc.shape[0]))
        else:
            h = jnp.full((N, 1), t)

        # Embed scalar node features to hidden dim
        h = nn.Dense(self.hidden_nf)(h)

        # Store initial positions
        x0 = x

        # Message-passing layers
        coords_range_layer = self.coords_range / self.n_layers
        for _ in range(self.n_layers):
            h, x = EGCLLayer(
                hidden_nf=self.hidden_nf,
                recurrent=self.recurrent,
                tanh=self.tanh,
                coords_range=coords_range_layer,
            )(h, x, self._senders, self._receivers)

        # Velocity = accumulated displacement
        vel = x - x0

        # Zero mean: enforce no net center-of-mass drift
        vel = vel - jnp.mean(vel, axis=0, keepdims=True)

        if flat_input:
            vel = vel.reshape(-1)
        return vel
