# config.py

import jax.numpy as jnp
import jax

class PerturbationConfig:
    def __init__(self, T, N, G, a, perturbation_strength, M, pulse_time, pulse_amp, pulse_width):
        """
        Initialize the configuration with user-defined perturbation parameters and optimizer settings.
        Precompute the harmonic indices (n_user, n_opt) and generate p_user based on half of M for each.

        Parameters:
        - T: Period of the perturbation
        - N: Number of time samples
        - G: Constant for gravitational calculation
        - a: Scaling constant
        - perturbation_strength: Strength of the perturbation
        - M: Total number of harmonics to be distributed between user and optimizer
        - pulse_time: Center time for the Gaussian pulse
        - pulse_amp: Amplitude of the Gaussian pulse
        - pulse_width: Width of the Gaussian pulse
        """
        
        # Primary parameters
        self.T = T
        self.N = N
        self.G = G
        self.a = a
        self.perturbation_strength = perturbation_strength
        self.M = M
        self.pulse_time = pulse_time
        self.pulse_amp = pulse_amp
        self.pulse_width = pulse_width

        # Calculated constants
        self.kappa = (8 * jnp.pi * self.G) / self.a
        self.C = self.a / (16 * jnp.pi * self.G)

        # Check if the pulse width is large enough to be detected given the sampling interval
        self.validate_pulse_width()

        # Time array
        self._t = jnp.linspace(0.001, T, N)

        # Create the full harmonic range up to M
        full_harmonic_range = jnp.arange(1, M + 1)

        # Assign alternating indices to user and optimizer harmonics
        self._n_user = full_harmonic_range[::2]  # Odd indices
        self._n_opt = full_harmonic_range[1::2]  # Even indices

        # Generate p_user as random perturbation parameters for user harmonics
        key = jax.random.PRNGKey(1)  # fixed seed for reproducibility
        self._p_user = jax.random.normal(key, shape=(2 * len(self._n_user),)) * 0.01 * self.perturbation_strength

    def validate_pulse_width(self):
        """Check if pulse width is sufficiently larger than the sampling interval."""
        # Calculate the sampling interval
        sampling_interval = self.T / self.N

        # Minimum acceptable pulse width (e.g., 10 times the sampling interval)
        min_pulse_width = 10 * sampling_interval

        # Raise a warning if pulse width is too narrow
        if self.pulse_amp > 0 and self.pulse_width < min_pulse_width:
            raise ValueError(
                f"Pulse width ({self.pulse_width}) is too narrow for the given sampling interval ({sampling_interval}). "
                f"Consider setting pulse_width to at least {min_pulse_width:.5f} to ensure the pulse is captured."
            )

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
              f"perturbation_strength={self.perturbation_strength}, M={self.M}")
        print(f"Harmonic Indices:\n n_user={self.n_user}, n_opt={self.n_opt}")

