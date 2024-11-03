# config.py

import jax.numpy as jnp
import jax

class PerturbationConfig:
    def __init__(self, T, N, C, perturbation_strength, M_user):
        """
        Initialize the configuration with user-defined perturbation parameters.
        Precompute the harmonic indices (n_user) and generate p_user based on M_user.

        Parameters:
        - T: Period of the perturbation
        - N: Number of time samples
        - C: Constant for action calculation
        - perturbation_strength: Strength of the perturbation
        - M_user: Number of user-defined harmonics
        """
        # Primary user perturbation parameters
        self.T = T
        self.N = N
        self.C = C
        self.perturbation_strength = perturbation_strength
        self.M_user = M_user

        # Precompute harmonic indices for user-defined harmonics
        self._n_user = jnp.arange(1, M_user + 1)

        # Generate p_user as random perturbation parameters (hidden from direct access)
        key = jax.random.PRNGKey(1)  # fixed seed for reproducibility
        self._p_user = jax.random.normal(key, shape=(2 * M_user,)) * 0.01

    @property
    def n_user(self):
        """Harmonic indices for user perturbation."""
        return self._n_user

    @property
    def p_user(self):
        """User-controlled perturbation parameters."""
        return self._p_user

    def display(self):
        """Optional method to display configuration details for debugging or logging."""
        print(f"PerturbationConfig:\n T={self.T}, N={self.N}, C={self.C}, "
              f"perturbation_strength={self.perturbation_strength}, M_user={self.M_user}")
        print(f"Harmonic Indices:\n n_user={self.n_user}")

