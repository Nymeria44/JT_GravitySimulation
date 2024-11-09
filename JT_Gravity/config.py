# config.py

import jax.numpy as jnp
import jax

class PerturbationConfig:
    def __init__(self, T, N, G, a, perturbation_strength, M_user, M_opt, pulse_time, pulse_amp, pulse_width):
        """
        Initialize the configuration with user-defined perturbation parameters and optimizer settings.
        Precompute the harmonic indices (n_user, n_opt) and generate p_user based on M_user and M_opt values.
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

        # Assign harmonics alternately to user and optimizer
        self._n_user, self._n_opt = self.assign_alternating_harmonics(M_user, M_opt)

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

    def assign_alternating_harmonics(self, M_user, M_opt):
        """
        Distribute harmonics alternately between user and optimizer up to the required counts.
        If one needs more harmonics than the other, assign the remaining harmonics in sequence.
        
        Returns:
        - n_user: Harmonic indices for the user
        - n_opt: Harmonic indices for the optimizer
        """
        total_harmonics = M_user + M_opt
        harmonics = jnp.arange(1, total_harmonics + 1)

        # Alternate assignment of harmonics
        n_user, n_opt = [], []
        for i, harmonic in enumerate(harmonics):
            if i % 2 == 0 and len(n_user) < M_user:
                n_user.append(harmonic)
            elif i % 2 == 1 and len(n_opt) < M_opt:
                n_opt.append(harmonic)

        # Assign any remaining harmonics to the list that needs more
        remaining = harmonics[len(n_user) + len(n_opt):]
        if len(n_user) < M_user:
            n_user.extend(remaining)
        else:
            n_opt.extend(remaining)

        return jnp.array(n_user), jnp.array(n_opt)

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
