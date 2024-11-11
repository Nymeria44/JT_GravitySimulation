import jax.numpy as jnp
from schwarzian import schwarzian_derivative

def compute_chiral_components(f_t, config):
    """
    Compute chiral components U(u) and V(v) from f(t).

    Parameters
    ----------
    f_t : jnp.ndarray
        Function values f(t) on time grid
    config : PerturbationConfig
        Configuration with coordinate grids

    Returns
    -------
    tuple
        (U(u), V(v)) arrays on respective grids
    """
    u = config.u
    v = config.v
    
    U_values = jnp.interp(u.flatten(), config.t, f_t).reshape(u.shape)
    V_values = jnp.interp(v.flatten(), config.t, f_t).reshape(v.shape)

    return U_values, V_values

def calculate_energy(p_opt, config):
    """
    Compute boundary energy from Schwarzian derivative.

    Parameters
    ----------
    p_opt : jnp.ndarray
        Optimizer-controlled coefficients
    config : PerturbationConfig
        System configuration

    Returns
    -------
    jnp.ndarray
        Boundary energy E(t)
    """
    schwarzian = schwarzian_derivative(p_opt, config)

    return -config.C * schwarzian

def calculate_beta(E_t, config):
    """
    Compute inverse temperature from boundary energy.

    Parameters
    ----------
    E_t : jnp.ndarray
        Boundary energy
    config : PerturbationConfig
        System configuration

    Returns
    -------
    jnp.ndarray
        Inverse temperature β(t)
    """
    return jnp.pi / jnp.sqrt(config.kappa * E_t)

def calculate_X_plus_minus(f_t, beta):
    """
    Compute X± coordinate from reparameterization.

    Parameters
    ----------
    f_t : jnp.ndarray
        Reparameterization function
    beta : jnp.ndarray
        Inverse temperature

    Returns
    -------
    jnp.ndarray
        X± coordinate values
    """
    return jnp.tan(jnp.pi * f_t / beta)

def calculate_dilaton_field(f_u, f_v, E, config):
    """
    Compute dilaton field Φ²(u,v) with reparameterized boundary.

    Parameters
    ----------
    f_u, f_v : jnp.ndarray
        Boundary reparameterization functions
    E : jnp.ndarray
        Boundary energy
    config : PerturbationConfig
        System configuration

    Returns
    -------
    jnp.ndarray
        Dilaton field Φ²(u,v)
    """
    beta = calculate_beta(E, config)
    X_plus = calculate_X_plus_minus(f_u, beta)
    X_minus = calculate_X_plus_minus(f_v, beta)

    numerator = 1 - config.kappa * E * X_plus * X_minus
    denominator = X_plus - X_minus + 1e-10

    return 1 + config.a * numerator / denominator
