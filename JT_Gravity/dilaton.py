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

    # TODO figure how to deal with negative sqrt.

    return beta_t

def calculate_X_plus_minus(f_t, beta):
    """
    Compute X^+ and X^- values for a given f(t) and beta.
    """
    return jnp.tan(jnp.pi * f_t / beta)

def calculate_dilaton_field(f_t, E_t, config):
    """
    Calculate the dilaton field Phi^2(u, v) for a single boundary horizon setup.

    Parameters:
    - f_t: Boundary reparameterization function for the single horizon
    - E_t: Energy for the boundary
    - config: Configuration object with constants
    
    Returns:
    - Phi_squared: The computed dilaton field.
    """
    # Calculate beta for the boundary
    beta = calculate_beta(E_t, config)

    # Compute X^+ and X^- based on a single boundary function
    X_plus = calculate_X_plus_minus(f_t, beta)
    X_minus = calculate_X_plus_minus(f_t, beta)  # For single horizon, X^- uses the same f_t

    # Calculate Phi^2(u, v) using the black hole coordinates
    numerator = 1 - config.kappa * E_t * X_plus * X_minus
    denominator = X_plus - X_minus + 1e-10  # Adding a small value to avoid division by zero
    Phi_squared = 1 + config.a * numerator / denominator

    return Phi_squared

