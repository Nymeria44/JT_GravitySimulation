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

def calculate_mu(E, config):
    """
    Calculate μ parameter from the energy E.
    
    In JT gravity, μ is proportional to the total energy (mass) in spacetime
    according to the relation μ = 8πGₙE, where Gₙ is Newton's constant.
    
    Parameters
    ----------
    E : jnp.ndarray
        Energy of the system
    config : PerturbationConfig
        System configuration containing Newton's constant G_N
        
    Returns
    -------
    jnp.ndarray
        μ parameter value(s)
    """
    return 8 * jnp.pi * config.G_N * E

def calculate_dilaton_field(f_u, f_v, E, config):
    """
    Compute dilaton field Φ(t,z) with reparameterized boundary.
    
    The dilaton field in JT gravity takes the form:
    Φ(t,z) = a - μ[f(t+z)·f(t-z)]/[f(t+z)-f(t-z)]
    
    Parameters
    ----------
    f_u : jnp.ndarray
        f(t+z) values
    f_v : jnp.ndarray
        f(t-z) values
    E : jnp.ndarray
        Boundary energy
    config : PerturbationConfig
        System configuration
        
    Returns
    -------
    jnp.ndarray
        Dilaton field Φ(t,z)
    """
    mu = calculate_mu(E, config)
    
    numerator = f_u * f_v
    denominator = f_u - f_v + 1e-10  # Small epsilon for numerical stability

    return config.a - mu * numerator / denominator

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
