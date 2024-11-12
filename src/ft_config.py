# ft_config.py

import jax.numpy as jnp
from typing import Dict, Any

from dilaton import (
    calculate_energy,
    compute_chiral_components,
    calculate_dilaton_field
)

################################################################################
# Main class
################################################################################

class FtOptimalConfig:
    def __init__(self, results: Dict[str, Dict[str, Any]], pert_config):
        """
        Initialize f(t) configuration and compute related fields.

        Parameters
        ----------
        results : dict
            Optimization results dictionary
        pert_config : PerturbationConfig
            Configuration for the perturbation parameters
        """
        self.select_best_optimizer(results)
        self.validate_arrays()
        self.compute_fields(pert_config)

################################################################################
# Selection Methods
################################################################################

    def select_best_optimizer(self, results: Dict[str, Dict[str, Any]]):
        """Select optimizer with action value closest to zero."""
        method = min(results["action_values"], 
                    key=lambda k: abs(results["action_values"][k]))
        
        print(f"Selected best optimizer: {method}")
        
        self._f_t = results["f_t"][method]
        self._parameters = results["optimized_params"][method]
        self._action_value = results["action_values"][method]
        self._computation_time = results["times_taken"][method]
        self._method = method

################################################################################
# Field Computation Methods
################################################################################

    def compute_fields(self, pert_config):
        """Compute all derived fields from f(t)."""
        self._f_u, self._f_v = compute_chiral_components(self._f_t, pert_config)
        self._energy = calculate_energy(self._parameters, pert_config)
        self._dilaton = calculate_dilaton_field(self._f_u, self._f_v, 
                                              self._energy, pert_config)

################################################################################
# Validation Methods
################################################################################

    def validate_arrays(self):
        """Verify array types."""
        if not isinstance(self._f_t, jnp.ndarray):
            raise TypeError("f_t must be a JAX numpy array")
        if not isinstance(self._parameters, jnp.ndarray):
            raise TypeError("parameters must be a JAX numpy array")

################################################################################
# Debug Methods
################################################################################

    def debug_info(self):
        """Print configuration details."""
        print("FtConfig Debug Information:")
        print(f"  Optimization method: {self._method}")
        print(f"  Action value: {self._action_value}")
        print(f"  Computation time: {self._computation_time}s")
        print(f"  f(t) shape: {self._f_t.shape}")
        print(f"  Parameters shape: {self._parameters.shape}")
        print(f"  Energy: {self._energy}")
        print(f"  Dilaton field shape: {self._dilaton.shape}")

################################################################################
# Properties
################################################################################

    @property
    def f_t(self):
        """Optimized f(t) solution array."""
        return self._f_t

    @property
    def parameters(self):
        """Optimization parameters."""
        return self._parameters

    @property
    def action_value(self):
        """Final action value."""
        return self._action_value

    @property
    def computation_time(self):
        """Optimization computation time."""
        return self._computation_time

    @property
    def method(self):
        """Optimization method used."""
        return self._method

    @property
    def f_u(self):
        """Chiral component u of f(t)."""
        return self._f_u
    
    @property
    def f_v(self):
        """Chiral component v of f(t)."""
        return self._f_v
    
    @property
    def energy(self):
        """Energy of the system."""
        return self._energy
    
    @property
    def dilaton(self):
        """Dilaton field configuration."""
        return self._dilaton
