import jax.numpy as jnp

from schwarzian import (
    schwarzian_derivative
)

def calculate_energy(p_opt, config):
    schwarzian = schwarzian_derivative(p_opt, config)
    E_t = -config.C * schwarzian
    return E_t

def calculate_beta(E_t, config):
    beta_t = jnp.pi / jnp.sqrt(config.kappa * E_t)
    return beta_t

def calculate_X_plus_minus(f_t, beta):
    return jnp.tan(jnp.pi * f_t / beta)

def calculate_dilaton_field(f_t, E_t, config):

    # Calculate beta from black hole mass E
    beta = jnp.pi / jnp.sqrt(config.kappa * E_t)

    # Calculate f(u) and f(v) using the optimized parameters

    # Compute X^+ and X^-
    X_plus = calculate_X_plus_minus(f_t, beta)
    X_minus = calculate_X_plus_minus(f_t, beta)

    # Calculate Phi^2(u, v)
    numerator = 1 - config.kappa * E_t * X_plus * X_minus
    denominator = X_plus - X_minus
    Phi_squared = 1 + config.a * numerator / denominator

    return Phi_squared
