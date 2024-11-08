# config.py

import jax.numpy as jnp
import jax

class PerturbationConfig:
    def __init__(self, T, N, G, a, perturbation_strength, M_user, M_opt, pulse_time, pulse_amp, pulse_width):
        """
        Initialize the configuration with user-defined perturbation parameters and optimizer settings.
        Precompute the harmonic indices (n_user, n_opt) and generate p_user based on M_user and M_opt values.

        Parameters:
        - T: Period of the perturbation
        - N: Number of time samples
        - G: Constant for gravitational calculation
        - a: Scaling constant
        - perturbation_strength: Strength of the perturbation
        - M_user: Number of harmonics for user control
        - M_opt: Number of harmonics for optimizer control
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
        self.M_user = M_user
        self.M_opt = M_opt
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

        # Create the harmonic range up to max(M_user + M_opt)
        total_M = self.M_user + self.M_opt
        full_harmonic_range = jnp.arange(1, total_M + 1)

        # Assign harmonics to user and optimizer up to their respective limits
        self._n_user = full_harmonic_range[::2][:self.M_user]  # Assign up to M_user harmonics
        self._n_opt = full_harmonic_range[1::2][:self.M_opt]  # Assign up to M_opt harmonics

        # If M_user or M_opt harmonics are exhausted, extend the other with remaining values
        if len(self._n_user) < self.M_user:
            remaining_user = full_harmonic_range[2 * len(self._n_user):][:self.M_user - len(self._n_user)]
            self._n_user = jnp.concatenate([self._n_user, remaining_user])

        if len(self._n_opt) < self.M_opt:
            remaining_opt = full_harmonic_range[2 * len(self._n_opt):][:self.M_opt - len(self._n_opt)]
            self._n_opt = jnp.concatenate([self._n_opt, remaining_opt])

        # Generate p_user as random perturbation parameters for user harmonics
        key = jax.random.PRNGKey(1)  # fixed seed for reproducibility
        self._p_user = jax.random.normal(key, shape=(2 * M_user,)) * 0.01 * self.perturbation_strength

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

    def debug_info(self):
            """Print detailed information for debugging."""
            print("PerturbationConfig Debug Information:")
            print(f"  T = {self.T}, N = {self.N}")
            print(f"  Gravitational constant (G) = {self.G}")
            print(f"  Stability parameter (a) = {self.a}")
            print(f"  Perturbation strength = {self.perturbation_strength}")
            print(f"  Number of harmonics (User) = {self.M_user}, Number of harmonics (Optimizer) = {self.M_opt}")
            print(f"  Pulse time = {self.pulse_time}, Pulse amplitude = {self.pulse_amp}, Pulse width = {self.pulse_width}")
            print(f"  Computed kappa = {self.kappa}, Computed C = {self.C}")
            print(f"  Harmonic indices (User) = {self._n_user}")
            print(f"  Harmonic indices (Optimizer) = {self._n_opt}")
            print(f"  User-controlled perturbation parameters (p_user) = {self._p_user}")
            print(f"  Time array (t) shape = {self._t.shape}")

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
