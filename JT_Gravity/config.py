# config.py

import jax.numpy as jnp
import jax

class PerturbationConfig:
    def __init__(self, T, Z, N, G, a, perturbation_strength, M_user, M_opt, pulse_time, pulse_amp, pulse_width):
        """
        Initialize the configuration with user-defined perturbation parameters and optimizer settings.
        Precompute the harmonic indices (n_user, n_opt) and generate p_user based on M_user and M_opt values.
        """
        
        # Primary parameters
        self.T = T
        self.Z = Z
        self.N = N
        self.G = G
        self.a = a
        self.perturbation_strength = perturbation_strength
        self.M_user = M_user
        self.M_opt = M_opt
        self.pulse_time = pulse_time
        self.pulse_amp = pulse_amp
        self.pulse_width = pulse_width

        # Calculate space-time volume
        self.setup_coordinates()

        # Calculated constants
        self.kappa = (8 * jnp.pi * self.G) / self.a
        self.C = self.a / (16 * jnp.pi * self.G)

        # Define time step dt for numerical integration
        self.dt = self.T / self.N  # Added line

        # Check if the pulse width is large enough to be detected given the sampling interval
        self.validate_pulse_width()

        # Assign harmonics alternately to user and optimizer
        self._n_user, self._n_opt = self.assign_harmonics(M_user, M_opt)

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

    def assign_harmonics(self, M_user, M_opt):
        """
        Distribute harmonics alternately between user and optimizer up to the required counts.
        If one list is exhausted, assign remaining harmonics to the other list.
        
        Returns:
        - n_user: Harmonic indices for the user
        - n_opt: Harmonic indices for the optimizer
        """
        n_user, n_opt = [], []
        total_harmonics = M_user + M_opt
        harmonics = jnp.arange(1, total_harmonics + 1)

        assign_to_user = True  # Flag to alternate assignments

        for harmonic in harmonics:
            if assign_to_user:
                if len(n_user) < M_user:
                    n_user.append(harmonic)
                    assign_to_user = False  # Next assignment to optimizer
                elif len(n_opt) < M_opt:
                    n_opt.append(harmonic)
            else:
                if len(n_opt) < M_opt:
                    n_opt.append(harmonic)
                    assign_to_user = True  # Next assignment to user
                elif len(n_user) < M_user:
                    n_user.append(harmonic)
        
        # Convert lists to JAX arrays
        return jnp.array(n_user), jnp.array(n_opt)


    def setup_coordinates(self):
        """
        Sets up the default, unchangeable time (_t) and spatial (_z) grids, and corresponding lightcone coordinates (_u, _v).
        """
        # Time array for t
        self._t = jnp.linspace(0.001, self.T, self.N)

        # Spatial array for z
        self._z = jnp.linspace(0, self.Z, self.N)

        # Generate meshgrids for t and z
        T_grid, Z_grid = jnp.meshgrid(self._t, self._z, indexing="ij")

        # Calculate u and v based on t and z
        self._u = T_grid + Z_grid
        self._v = T_grid - Z_grid

    def to_bulk_coordinates(self, u, v):
        """
        Convert lightcone coordinates (u, v) to bulk coordinates (t, z).
        
        Parameters:
        - u (float or array): Lightcone coordinate u.
        - v (float or array): Lightcone coordinate v.
        
        Returns:
        - tuple: (t, z), where t = (u + v) / 2 and z = (u - v) / 2.
        """
        t = (u + v) / 2
        z = (u - v) / 2
        return t, z

    def to_lightcone_coordinates(self, t, z):
        """
        Convert bulk coordinates (t, z) to lightcone coordinates (u, v).
        
        Parameters:
        - t (float or array): Bulk time coordinate.
        - z (float or array): Bulk spatial coordinate.
        
        Returns:
        - tuple: (u, v), where u = t + z and v = t - z.
        """
        u = t + z
        v = t - z
        return u, v

    def debug_info(self):
        """Print detailed information for debugging."""
        print("PerturbationConfig Debug Information:")
        print(f"  T = {self.T}, Z = {self.Z}, N = {self.N}")
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
        print(f"  Spatial array (z) shape = {self._z.shape}")
        print(f"  Lightcone coordinate arrays (u, v) shapes = {self._u.shape}, {self._v.shape}")

    @property
    def t(self):
        """Time array for the perturbation."""
        return self._t

    @property
    def z(self):
        """Spatial array for the perturbation (z)."""
        return self._z

    @property
    def u(self):
        """Lightcone coordinate array (u)."""
        return self._u

    @property
    def v(self):
        """Lightcone coordinate array (v)."""
        return self._v

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

