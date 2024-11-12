# config.py

import jax.numpy as jnp
import jax

################################################################################
# Main class
################################################################################

class PerturbationConfig:
    def __init__(self, T, Z, N, G, a, perturbation_strength, M_user, M_opt, 
                 pulse_time, pulse_amp, pulse_width):
        """
        Initialize perturbation configuration and optimizer settings.

        Parameters
        ----------
        T : float
            Total time period
        Z : float
            Spatial extent
        N : int
            Number of grid points
        G : float
            Gravitational constant
        a : float
            Stability parameter
        perturbation_strength : float
            Overall strength of perturbations
        M_user : int
            Number of user-controlled harmonics
        M_opt : int
            Number of optimizer-controlled harmonics
        pulse_time : float
            Center time of Gaussian pulse
        pulse_amp : float
            Amplitude of Gaussian pulse
        pulse_width : float
            Width of Gaussian pulse
        """
        # Coordinate system parameters
        self.T = T
        self.Z = Z
        self.N = N
        
        # Physical parameters
        self.G = G
        self.a = a

        # Fourier series parameters
        self.perturbation_strength = perturbation_strength
        self.M_user = M_user
        self.M_opt = M_opt

        # Gaussian pulse parameters
        self.pulse_time = pulse_time
        self.pulse_amp = pulse_amp
        self.pulse_width = pulse_width

        # Setup derived quantities
        self.setup_coordinates()
        self.kappa = (8 * jnp.pi * self.G) / self.a
        self.C = self.a / (16 * jnp.pi * self.G)
        self.dt = self.T / self.N

        self.validate_pulse_width()
        self._n_user, self._n_opt = self.assign_harmonics(M_user, M_opt)
        
        # Generate random user perturbation parameters
        key = jax.random.PRNGKey(1)
        self._p_user = jax.random.normal(key, shape=(2 * M_user,)) * 0.01 * self.perturbation_strength

################################################################################
# Coordinate Methods
################################################################################

    def setup_coordinates(self):
        """
        Initialize coordinate grids and lightcone coordinates.
        """
        # Setup basic coordinate grids
        self._t = jnp.linspace(0.001, self.T, self.N)
        self._z = jnp.linspace(0, self.Z, self.N)

        # Create meshgrids for lightcone coordinates
        T_grid, Z_grid = jnp.meshgrid(self._t, self._z, indexing="ij")
        self._u = T_grid + Z_grid
        self._v = T_grid - Z_grid

    def to_bulk_coordinates(self, u, v):
        """
        Convert lightcone to bulk coordinates.

        Parameters
        ----------
        u, v : jnp.ndarray
            Lightcone coordinates

        Returns
        -------
        tuple
            Bulk coordinates (t, z)
        """
        return (u + v) / 2, (u - v) / 2

    def to_lightcone_coordinates(self, t, z):
        """
        Convert bulk to lightcone coordinates.

        Parameters
        ----------
        t, z : jnp.ndarray
            Bulk coordinates

        Returns
        -------
        tuple
            Lightcone coordinates (u, v)
        """
        return t + z, t - z

################################################################################
# Validation and Setup Methods
################################################################################

    def validate_pulse_width(self):
        """
        Verify pulse width against sampling interval.

        Raises
        ------
        ValueError
            If pulse width is too narrow for grid resolution
        """
        sampling_interval = self.T / self.N
        min_pulse_width = 10 * sampling_interval

        if self.pulse_amp > 0 and self.pulse_width < min_pulse_width:
            raise ValueError(
                f"Pulse width ({self.pulse_width}) too narrow for sampling interval ({sampling_interval}). "
                f"Minimum recommended width: {min_pulse_width:.5f}"
            )

    def assign_harmonics(self, M_user, M_opt):
        """
        Distribute harmonics between user and optimizer.

        Parameters
        ----------
        M_user : int
            Number of user harmonics
        M_opt : int
            Number of optimizer harmonics

        Returns
        -------
        tuple
            (n_user, n_opt) harmonic indices arrays
        """
        n_user, n_opt = [], []
        harmonics = jnp.arange(1, M_user + M_opt + 1)
        assign_to_user = True

        for harmonic in harmonics:
            if assign_to_user:
                if len(n_user) < M_user:
                    n_user.append(harmonic)
                    assign_to_user = False
                elif len(n_opt) < M_opt:
                    n_opt.append(harmonic)
            else:
                if len(n_opt) < M_opt:
                    n_opt.append(harmonic)
                    assign_to_user = True
                elif len(n_user) < M_user:
                    n_user.append(harmonic)

        return jnp.array(n_user), jnp.array(n_opt)

################################################################################
# Debug Methods
################################################################################

    def debug_info(self):
        """
        Print configuration details for debugging.
        """
        print("PerturbationConfig Debug Information:")
        print(f"  T = {self.T}, Z = {self.Z}, N = {self.N}")
        print(f"  G = {self.G}, a = {self.a}")
        print(f"  perturbation_strength = {self.perturbation_strength}")
        print(f"  M_user = {self.M_user}, M_opt = {self.M_opt}")
        print(f"  pulse_time = {self.pulse_time}, amp = {self.pulse_amp}, width = {self.pulse_width}")
        print(f"  kappa = {self.kappa}, C = {self.C}")
        print(f"  n_user = {self._n_user}, n_opt = {self._n_opt}")
        print(f"  p_user = {self._p_user}")
        print(f"  Grid shapes: t={self._t.shape}, z={self._z.shape}, u={self._u.shape}, v={self._v.shape}")

################################################################################
# Properties
################################################################################

    @property
    def t(self):
        """Time coordinate array."""
        return self._t

    @property
    def z(self):
        """Spatial coordinate array."""
        return self._z

    @property
    def u(self):
        """Lightcone u-coordinate array."""
        return self._u

    @property
    def v(self):
        """Lightcone v-coordinate array."""
        return self._v

    @property
    def n_user(self):
        """User harmonic indices."""
        return self._n_user

    @property
    def n_opt(self):
        """Optimizer harmonic indices."""
        return self._n_opt

    @property
    def p_user(self):
        """User perturbation parameters."""
        return self._p_user
