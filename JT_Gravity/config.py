# config.py

import jax.numpy as jnp
import jax

class PerturbationConfig:
    def __init__(self, T, N, C, perturbation_strength, M_user, M_opt):
        """
        Initialize the configuration with user-defined perturbation parameters and optimizer settings.
        Precompute the harmonic indices (n_user, n_opt) and generate p_user based on M_user.

        Parameters:
        - T: Period of the perturbation
        - N: Number of time samples
        - C: Constant for action calculation
        - perturbation_strength: Strength of the perturbation
        - M_user: Number of user-defined harmonics
        - M_opt: Number of optimizer-controlled harmonics
        """
        # Primary parameters
        self.T = T
        self.N = N
        self.C = C
        self.perturbation_strength = perturbation_strength
        self.M_user = M_user
        self.M_opt = M_opt

        # Time array
        self._t = jnp.linspace(0.001, T, N)

        # Precompute harmonic indices for user-defined and optimizer-controlled harmonics
        self._n_user = jnp.arange(1, M_user + 1)
        self._n_opt = jnp.arange(M_user + 1, M_user + M_opt + 1)

        # Generate p_user as random perturbation parameters (hidden from direct access)
        key = jax.random.PRNGKey(1)  # fixed seed for reproducibility
        self._p_user = jax.random.normal(key, shape=(2 * M_user,)) * 0.01 * self.perturbation_strength

    @property
    def t(self):
        """Time array for the perturbation."""
        return self._t

    @property
    def n_user(self):
        """Harmonic indices for user perturbation."""
        return self._n_user

    @property
    def n_opt(self):
        """Harmonic indices for optimizer-controlled perturbation."""
        return self._n_opt

    @property
    def p_user(self):
        """User-controlled perturbation parameters."""
        return self._p_user

    def display(self):
        """Optional method to display configuration details for debugging or logging."""
        print(f"PerturbationConfig:\n T={self.T}, N={self.N}, C={self.C}, "
              f"perturbation_strength={self.perturbation_strength}, M_user={self.M_user}, M_opt={self.M_opt}")
        print(f"Harmonic Indices:\n n_user={self.n_user}, n_opt={self.n_opt}")
