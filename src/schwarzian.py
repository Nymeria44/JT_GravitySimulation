# schwarzianJAX.py

import jax
import jax.numpy as jnp

from optimisations import (
    run_bfgs_optimization,
    run_adam_optimization,
    run_optax_adam_optimization,
    run_newtons_method,
    run_hessian_optimization,
    run_yogi_optimization,
    run_lbfgs_optimization,
    run_adabelief_optimization,
)

from config import PerturbationConfig  # Import the configuration class


################################################################################
# Enforcing Monotonicity Constraints
################################################################################
def enforce_monotone_constraints(coeffs_sin, coeffs_cos, T, n, threshold=0.999, scale_target=0.998):
    """
    Projects Fourier coefficients to ensure monotonicity: f'(t) > 0
    Using vectorized operations for better performance
    
    Parameters
    ----------
    coeffs_sin : jnp.ndarray
        Sine coefficients
    coeffs_cos : jnp.ndarray
        Cosine coefficients
    T : float
        Period
    n : jnp.ndarray
        Array of harmonic indices
    threshold : float, optional
        Threshold for maximum derivative (default: 0.99)
    scale_target : float, optional
        Target scale factor when threshold is exceeded (default: 0.98)
    
    Returns
    -------
    tuple
        Projected (coeffs_sin, coeffs_cos)
    """
    # Calculate frequency factor
    omega_n = (2 * jnp.pi / T) * n
    
    # Vectorized calculation of maximum derivative contribution
    amplitudes = jnp.sqrt(coeffs_cos**2 + coeffs_sin**2)
    max_derivative = jnp.sum(omega_n * amplitudes)
    
    # Apply scaling with configurable threshold
    scale_factor = jnp.where(
        max_derivative >= threshold,
        scale_target / max_derivative,
        1.0
    )
    
    # Scale both coefficient arrays at once
    return coeffs_sin * scale_factor, coeffs_cos * scale_factor

################################################################################
# Core Function Calculations (f(t) and derivatives)
################################################################################

def calculate_delta_f(t, p, M, T, n, order=0, pulse_time=None, pulse_amp=0, pulse_width=0.00):
    """
    Calculate Fourier series perturbation with monotonicity constraints
    
    Parameters
    ----------
    t : jnp.ndarray
        Time points
    p : jnp.ndarray
        Fourier coefficients (sin and cos combined)
    M : int
        Number of terms in each series
    T : float
        Period
    n : jnp.ndarray
        Harmonic indices
    order : int
        Derivative order (0=function, 1=first derivative, etc.)
    pulse_time : float, optional
        Center time for Gaussian pulse
    pulse_amp : float, optional
        Amplitude of Gaussian pulse
    pulse_width : float, optional
        Width of Gaussian pulse
    
    Returns
    -------
    jnp.ndarray
        Perturbation or its derivative
    """
    delta_f = 0

    if p is not None and p.size > 0:
        # Split coefficients
        sin_coeffs = p[:M]
        cos_coeffs = p[M:]
        
        # Project coefficients if calculating f(t) or f'(t)
        if order <= 1:
            sin_coeffs, cos_coeffs = enforce_monotone_constraints(sin_coeffs, cos_coeffs, T, n)
        
        # Calculate frequency factor
        omega = 2 * jnp.pi / T
        factor = (omega * n) ** order
        
        # Calculate terms based on derivative order
        if order % 4 == 0:
            sin_terms = jnp.sin(omega * n[:, None] * t[None, :])
            cos_terms = jnp.cos(omega * n[:, None] * t[None, :])
        elif order % 4 == 1:
            sin_terms = jnp.cos(omega * n[:, None] * t[None, :])
            cos_terms = -jnp.sin(omega * n[:, None] * t[None, :])
        elif order % 4 == 2:
            sin_terms = -jnp.sin(omega * n[:, None] * t[None, :])
            cos_terms = -jnp.cos(omega * n[:, None] * t[None, :])
        else:  # order % 4 == 3
            sin_terms = -jnp.cos(omega * n[:, None] * t[None, :])
            cos_terms = jnp.sin(omega * n[:, None] * t[None, :])
        
        delta_f = jnp.dot(sin_coeffs * factor, sin_terms) + jnp.dot(cos_coeffs * factor, cos_terms)

    # Add Gaussian pulse if specified
    if pulse_time is not None and pulse_amp != 0 and pulse_width != 0:
        t_diff = t - pulse_time
        gaussian_base = pulse_amp / jnp.sqrt(2 * jnp.pi * pulse_width**2)
        
        if order == 0:
            gaussian_pulse = gaussian_base * jnp.exp(-t_diff**2 / (2 * pulse_width**2))
        elif order == 1:
            gaussian_pulse = gaussian_base * (-t_diff / pulse_width**2) * jnp.exp(-t_diff**2 / (2 * pulse_width**2))
        elif order == 2:
            gaussian_pulse = gaussian_base * ((t_diff**2 - pulse_width**2) / pulse_width**4) * jnp.exp(-t_diff**2 / (2 * pulse_width**2))
        elif order == 3:
            gaussian_pulse = gaussian_base * (-t_diff * (t_diff**2 - 3 * pulse_width**2) / pulse_width**6) * jnp.exp(-t_diff**2 / (2 * pulse_width**2))
        else:
            raise ValueError("Higher derivatives not implemented for Gaussian pulse")
            
        delta_f += gaussian_pulse
    
    return delta_f

def calculate_f(p_opt, config: PerturbationConfig, order=0):
    """
    Calculate f(t) = t + δf(t) and its derivatives
    
    Parameters
    ----------
    p_opt : jnp.ndarray
        Optimizer-controlled coefficients
    config : PerturbationConfig
        Configuration parameters
    order : int
        Derivative order
    
    Returns
    -------
    jnp.ndarray
        Function or derivative values
    """
    # Calculate perturbations
    delta_f_user = calculate_delta_f(
        config._t, config.p_user, 
        config.M_user, config.T, 
        config.n_user, order=order,
        pulse_time=config.pulse_time,
        pulse_amp=config.pulse_amp,
        pulse_width=config.pulse_width
    )
    
    delta_f_opt = calculate_delta_f(
        config._t, p_opt, 
        config.M_opt, config.T, 
        config.n_opt, order=order
    )
    
    # Add baseline term (t for f(t), 1 for f'(t), 0 for higher derivatives)
    baseline = jnp.where(order == 0, 
                        config._t,
                        jnp.where(order == 1, 1.0, 0.0))
    
    result = baseline + delta_f_user + delta_f_opt
    
    if order == 1:
        min_deriv = jnp.min(result)
        jax.lax.cond(
            min_deriv <= 0,
            lambda _: jax.debug.print("WARNING: Non-monotonic behavior detected, min f'(t) = {x}", x=min_deriv),
            lambda _: None,
            None
        )
    
    return result

################################################################################
# Schwarzian Action Computation
################################################################################

def schwarzian_derivative(p_opt, config: PerturbationConfig):
    """
    Compute Schwarzian derivative S(f, t).

    Parameters
    ----------
    p_opt : jnp.ndarray
        Optimizer-controlled Fourier coefficients
    config : PerturbationConfig
        User and optimizer parameters

    Returns
    -------
    jnp.ndarray
        Schwarzian derivative S(f) = f'''/f' - (3/2)(f''/f')²
    """
    fp = calculate_f(p_opt, config, order=1)
    fpp = calculate_f(p_opt, config, order=2)
    fppp = calculate_f(p_opt, config, order=3)

    S = fppp / fp - 1.5 * (fpp / fp) ** 2
    return S

def action_to_minimize(p_opt, config: PerturbationConfig):
    """
    Compute Schwarzian action via integration.
    This is the objective function for optimization.

    Parameters
    ----------
    p_opt : jnp.ndarray
        Optimizer-controlled Fourier coefficients
    config : PerturbationConfig
        System configuration parameters

    Returns
    -------
    float
        Action value -C∫S(f)dt
    """
    S = schwarzian_derivative(p_opt, config)
    return -config.C * jax.scipy.integrate.trapezoid(S, config._t)

################################################################################
# Optimization Functionality
################################################################################

def run_optimizations(action_to_minimize, p_initial, config, verbose=False):
    """
    Execute optimization methods based on configuration.

    Parameters
    ----------
    action_to_minimize : callable
        Target function for optimization
    p_initial : jnp.ndarray
        Initial parameter guess
    config : dict
        Optimizer configuration flags
    verbose : bool, default=False
        Whether to print optimization progress

    Returns
    -------
    dict
        Results for each optimizer including:
        - optimized_params
        - action_values
        - times_taken
    """
    optimization_methods = [
        ("BFGS", run_bfgs_optimization, (action_to_minimize, p_initial, verbose)) if config["BFGS"] else None,
        ("Adam (JAX)", run_adam_optimization, (action_to_minimize, p_initial, verbose)) if config["Adam (JAX)"] else None,
        ("Adam (Optax)", run_optax_adam_optimization, (action_to_minimize, p_initial, verbose)) if config["Adam (Optax)"] else None,
        ("Yogi", run_yogi_optimization, (action_to_minimize, p_initial, verbose)) if config["Yogi"] else None,
        ("LBFGS", run_lbfgs_optimization, (action_to_minimize, p_initial, verbose)) if config["LBFGS"] else None,
        ("AdaBelief", run_adabelief_optimization, (action_to_minimize, p_initial, verbose)) if config["AdaBelief"] else None,
        ("Newton's Method", run_newtons_method, (action_to_minimize, p_initial, verbose)) if config["Newton's Method"] else None,
        ("Hessian-based Optimization", run_hessian_optimization, (action_to_minimize, p_initial, verbose)) if config["Hessian-based Optimization"] else None
    ]

    optimization_methods = [method for method in optimization_methods if method is not None]

    results = {
        "optimized_params": {},
        "action_values": {},
        "times_taken": {}
    }

    for method_name, optimization_function, args in optimization_methods:
        if verbose:
            print(f"\n{'-'*60}")
            print(f"Running {method_name}")
            print(f"{'-'*60}")
        
        p_optimal, action_value, time_taken = optimization_function(*args)
        
        # Skip storing NaN results
        if jnp.isnan(action_value):
            if verbose:
                print(f"{method_name} failed to converge.")
            continue
            
        results["optimized_params"][method_name] = p_optimal
        results["action_values"][method_name] = action_value
        results["times_taken"][method_name] = time_taken
        
        if verbose:
            print(f"{method_name} completed in {time_taken:.4f}s (Action: {action_value:.6f})")

    return results

def reparameterise_ft(action_to_minimize, p_initial, config, pert_config, verbose=False):
    """
    Optimize f(t) reparameterization with full results.

    Parameters
    ----------
    action_to_minimize : callable
        Target function for optimization
    p_initial : jnp.ndarray
        Initial parameter guess
    config : dict
        Optimizer configuration flags
    pert_config : PerturbationConfig
        Parameters for f(t) calculation
    verbose : bool, default=False
        Whether to print optimization progress

    Returns
    -------
    dict
        Complete optimization results including:
        - optimized_params
        - action_value
        - time_taken
        - f_t values
    """
    if verbose:
        print("\nStarting reparametrization optimization...")
    
    # Run optimizations and get base results with verbose parameter
    results = run_optimizations(action_to_minimize, p_initial, config, verbose=verbose)

    # Calculate f(t) for each optimizer's result and add to results
    results["f_t"] = {}
    if verbose:
        print("\n" + "-"*60)
        print("Calculating f(t) for all optimizers...")
        print("-"*60)
        
    try:
        for method_name, p_optimal in results["optimized_params"].items():
            f_t_values = calculate_f(p_optimal, pert_config, order=0)
            results["f_t"][method_name] = f_t_values
            
        if verbose:
            print("All f(t) calculations completed successfully")
            print("-"*60)
    except Exception as e:
        if verbose:
            print(f"Error during f(t) calculations: {str(e)}")
            print("-"*60)

    if verbose:
        print("\nReparametrization optimization completed.")

    return results
