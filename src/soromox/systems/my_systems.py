__all__ = ["PlanarPCS_simple", "PlanarPCS_simple_modified"]
import equinox as eqx
import jax
from jax import Array, lax, vmap
from jax import numpy as jnp
import numpy as onp
from typing import Callable, Dict, Tuple, Optional, ClassVar

from soromox.systems.pcs.planar_pcs import PlanarPCS
from soromox.utils.integration import scale_gaussian_quadrature
import soromox.utils.lie_algebra as lie


# Simplified planar PCS model, without Coriolis and centrifugal effects
class PlanarPCS_simple(PlanarPCS):
    """
    Planar Piecewise Constant Strain (PCS) model for 2D soft continuum robots. Simplified 
    model (no Coriolis force). Inherits from PlanarPCS class.
    """

    @eqx.filter_jit
    def forward_dynamics(
        self, t: Array, y: Array, actuation_args: tuple | None = None
    ) -> Array:
        """
        Forward dynamics function. ! No Coriolis/centrifugal effects !

        Args:
            t (Array): Current time.
            y (Array): State vector containing configuration and velocity.
                Shape is (2 * num_strains,).
            actuation_args (Tuple, optional): Additional arguments for the actuation mapping function.
                Default is None.

        Returns:
            yd (Array): Time derivative of the state vector.
        """
        # Split the state vector into configuration and velocity
        q, qd = jnp.split(y, 2)

        # split the actuation arguments if provided
        if actuation_args is None:
            u, tau_ext = None, None
        elif len(actuation_args) == 1:
            u = actuation_args[0]
            tau_ext = None
        elif len(actuation_args) == 2:
            u, tau_ext = actuation_args
        else:
            raise ValueError("actuation_args must be a tuple of length 1 or 2.")

        if u is None:
            u = jnp.zeros((self.num_actuators,))
        if tau_ext is None:
            tau_ext = jnp.zeros((q.shape[-1],))

        B, Cqd, G = self._active_quadrature_forward_dynamics_terms(q, qd)
        tau_el = self.elastic_force(q)
        tau_u = self.actuation_force(q, u)

        rhs = tau_u + tau_ext - G - tau_el - self.damping_matrix(q) @ qd
        qdd = jnp.linalg.solve(B, rhs)

        yd = jnp.concatenate([qd, qdd])

        return yd
