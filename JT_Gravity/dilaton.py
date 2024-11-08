import jax.numpy as jnp
from schwarzian import (
    schwarzian_derivative
)

def calculate_energy(p_opt, config):
    """
    Calculate boundary energy using the Schwarzian derivative.
    """
    schwarzian = schwarzian_derivative(p_opt, config)
    E_t = -config.C * schwarzian
    return E_t

def calculate_beta(E_t, config):
    """
    Calculate inverse temperature beta from the boundary energy E_t.
    """
    beta_t = jnp.pi / jnp.sqrt(config.kappa * E_t)
    return beta_t

def calculate_X_plus_minus(f_t, beta):
    """
    Compute the X^+ or X^- value for given f(t) and beta, reflecting reparameterization.
    """
    return jnp.tan(jnp.pi * f_t / beta)

def calculate_dilaton_field(f_u, f_v, E, config):
    """
    Calculate the dilaton field Phi^2(u, v) for a reparameterized boundary horizon setup.

    Parameters:
    - f_u, f_v: Boundary reparameterization functions for each boundary
    - E: Energy for the boundary, influenced by the Schwarzian derivative
    - config: Configuration object with constants

    Returns:
    - Phi_squared: The computed dilaton field with reparameterized boundary.
    """
    # Calculate beta for the boundary
    beta = calculate_beta(E, config)

    # Compute X^+ and X^- using the reparameterized time
    X_plus = calculate_X_plus_minus(f_u, beta)
    X_minus = calculate_X_plus_minus(f_v, beta)

    # Calculate Phi^2(u, v) using the black hole coordinates with reparameterization
    numerator = 1 - config.kappa * E * X_plus * X_minus
    denominator = X_plus - X_minus + 1e-10  # Small offset to avoid division by zero
    Phi_squared = 1 + config.a * numerator / denominator

    return Phi_squared
