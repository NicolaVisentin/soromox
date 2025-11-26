__all__ = ["PlanarPCS_simple", "PlanarPCS_simple_modified"]
import equinox as eqx
import jax
from jax import Array, lax, vmap
from jax import numpy as jnp
import numpy as onp
from typing import Callable, Dict, Tuple, Optional, ClassVar

from soromox.systems.planar_pcs import PlanarPCS
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
        self, t: Array, y: Array, actuation_args: Optional[Tuple] = None
    ) -> Array:
        """
        Forward dynamics function. ! No Coriolis effect !

        Args:
            t (Array): Current time.
            y (Array): State vector containing configuration and velocity.
                Shape is (2 * num_strains,).
            actuation_args (Tuple, optional): Additional arguments for the actuation mapping function.
                Default is None.
        Returns:
            yd: Time derivative of the state vector.
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

        Xs_scaled, Ws_scaled = vmap(
            scale_gaussian_quadrature, in_axes=(None, None, 0, 0)
        )(self.Xs, self.Ws, self.L_cum[:-1], self.L_cum[1:])

        chi_ps = self.forward_kinematics_batched(q, Xs_scaled.flatten())    # [th, x, y] for all quadrature points. Shape (num_segments*(order_gauss+2), 3)
        g_ps = vmap(lie.exp_SE2)(chi_ps.reshape(-1, 3))                     # SE(2) matrix for all quadrature points. Shape (num_segments*(order_gauss+2), 3, 3)
        g_ps = g_ps.reshape(self.num_segments, self.num_gauss_points, 3, 3) # SE(2) matrix for all quadr points "reshaped" to (num_segments, order_gauss+2, 3, 3)

        J_ps, Jd_ps = self._J_Jd_local_batched(q, qd, Xs_scaled.flatten())
        J_ps = J_ps.reshape(self.num_segments, self.num_gauss_points, *J_ps.shape[1:])
        Jd_ps = Jd_ps.reshape(
            self.num_segments, self.num_gauss_points, *Jd_ps.shape[1:]
        )

        def dynamical_matrices_i(i: Array) -> Tuple[Array, Array, Array]:
            M_i = self._local_mass_matrix(i)

            def dynamical_matrices_ij(j: Array) -> Tuple[Array, Array, Array]:
                Ws_ij = Ws_scaled[i][j]
                g_ij = g_ps[i, j]
                J_ij = J_ps[i, j]

                Ad_g_inv_ij = lie.Adjoint_g_inv_SE2(g_ij)

                B_ij = Ws_ij * J_ij.T @ M_i @ J_ij
                G_ij = -Ws_ij * J_ij.T @ M_i @ Ad_g_inv_ij @ self.g

                return B_ij, G_ij

            return vmap(dynamical_matrices_ij)(jnp.arange(1, self.num_gauss_points - 1))

        B_blocks_tot, G_blocks_tot = vmap(dynamical_matrices_i)(
            jnp.arange(self.num_segments)
        )

        B_full = jnp.sum(B_blocks_tot, axis=(0, 1))
        G_full = jnp.sum(G_blocks_tot, axis=(0, 1))

        B = self.B_xi.T @ B_full @ self.B_xi
        G = self.B_xi.T @ G_full
        D = self.damping_matrix(q)
        tau_el = self.elastic_force(q)
        tau_u = self.actuation_force(q, u)

        B_inv = jnp.linalg.inv(B)  # Inverse of the inertia matrix
        qdd = B_inv @ (
            tau_u + tau_ext - G - tau_el - D @ qd
        )  # Compute the acceleration

        yd = jnp.concatenate([qd, qdd])

        return yd


# Modified PlanarPCS_simple for testing optimization with T_in and T_out inside the robot's dynamics
class PlanarPCS_simple_modified(PlanarPCS_simple):
    """
    Approximator model f_approx = T_out(f_pcs(T_in(y,yd))). It's a simplified planar PCS
    model (no Coriolis force), modified to have a dynamics ydd_hat = T_out(f_pcs(T_in(y,yd))).
    Inherits from PlanarPCS_simple class.

    Notation
    --------
    z : PCS state z = [q, qd]
    r : Approzimator state r = [y, yd]

    Additional attributes
    ---------------------
    A : Array
        Transform q = A*y + c.
    c : Array
        Transform q = A*y + c.
    """
    A_Tin: Array = eqx.field()
    c_Tin: Array = eqx.field()

    def __init__(self, *args, A=None, c=None, **kwargs):
        # Prima inizializzi i campi frozen
        object.__setattr__(self, "A_Tin", A)
        object.__setattr__(self, "c_Tin", c)

        # Poi chiami il costruttore della classe base
        super().__init__(*args, **kwargs)

    @eqx.filter_jit
    def forward_dynamics(
        self, t: Array, r: Array, actuation_args: Optional[Tuple] = None, 
    ) -> Array:
        """
        Forward dynamics function. ! No Coriolis effect !

        Args:
            t (Array): Current time.
            r (Array): State vector of the approximator r = [y, yd].
            actuation_args (Tuple, optional): Additional arguments for the actuation mapping function.
                Default is None.
        Returns:
            rd: Time derivative of the state vector.
        """
        # Split the state vector into configuration and velocity
        y, yd = jnp.split(r, 2)

        # Apply T_in
        q = self.A_Tin @ y + self.c_Tin
        qd = self.A_Tin @ yd

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

        Xs_scaled, Ws_scaled = vmap(
            scale_gaussian_quadrature, in_axes=(None, None, 0, 0)
        )(self.Xs, self.Ws, self.L_cum[:-1], self.L_cum[1:])

        chi_ps = self.forward_kinematics_batched(q, Xs_scaled.flatten())    # [th, x, y] for all quadrature points. Shape (num_segments*(order_gauss+2), 3)
        g_ps = vmap(lie.exp_SE2)(chi_ps.reshape(-1, 3))                     # SE(2) matrix for all quadrature points. Shape (num_segments*(order_gauss+2), 3, 3)
        g_ps = g_ps.reshape(self.num_segments, self.num_gauss_points, 3, 3) # SE(2) matrix for all quadr points "reshaped" to (num_segments, order_gauss+2, 3, 3)

        J_ps, Jd_ps = self._J_Jd_local_batched(q, qd, Xs_scaled.flatten())
        J_ps = J_ps.reshape(self.num_segments, self.num_gauss_points, *J_ps.shape[1:])
        Jd_ps = Jd_ps.reshape(
            self.num_segments, self.num_gauss_points, *Jd_ps.shape[1:]
        )

        def dynamical_matrices_i(i: Array) -> Tuple[Array, Array, Array]:
            M_i = self._local_mass_matrix(i)

            def dynamical_matrices_ij(j: Array) -> Tuple[Array, Array, Array]:
                Ws_ij = Ws_scaled[i][j]
                g_ij = g_ps[i, j]
                J_ij = J_ps[i, j]

                Ad_g_inv_ij = lie.Adjoint_g_inv_SE2(g_ij)

                B_ij = Ws_ij * J_ij.T @ M_i @ J_ij
                G_ij = -Ws_ij * J_ij.T @ M_i @ Ad_g_inv_ij @ self.g

                return B_ij, G_ij

            return vmap(dynamical_matrices_ij)(jnp.arange(1, self.num_gauss_points - 1))

        B_blocks_tot, G_blocks_tot = vmap(dynamical_matrices_i)(
            jnp.arange(self.num_segments)
        )

        B_full = jnp.sum(B_blocks_tot, axis=(0, 1))
        G_full = jnp.sum(G_blocks_tot, axis=(0, 1))

        B = self.B_xi.T @ B_full @ self.B_xi
        G = self.B_xi.T @ G_full
        D = self.damping_matrix(q)
        tau_el = self.elastic_force(q)
        tau_u = self.actuation_force(q, u)

        B_inv = jnp.linalg.inv(B)  # Inverse of the inertia matrix
        qdd = B_inv @ (
            tau_u + tau_ext - G - tau_el - D @ qd
        )  # Compute the acceleration

        # Apply T_out
        ydd = jnp.linalg.pinv(self.A_Tin) @ qdd
        rd = jnp.concatenate([yd, ydd])

        return rd