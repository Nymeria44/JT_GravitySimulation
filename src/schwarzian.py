# schwarzianJAX.py

import jax
import jax.numpy as jnp

# Custom Imports from `optimisations.py`
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

def calculate_delta_g(t, p, M, T, n, order=0, pulse_time=None, pulse_amp=0, pulse_width=0.00):
    """
    Calculate the Fourier series terms / Gaussian pulse and their derivatives for g(t)

    Parameters:
    - t: time variable (1D array)
    - p: Fourier coefficients (sin and cos combined) (1D array)
    - M: Number of terms in each series (sin and cos)
    - T: Period
    - n: Array of harmonic indices (1D array)
    - order: The derivative order (0 for the function itself, 1 for first derivative, etc.)
    - pulse_time: Center time for the Gaussian pulse (optional)
    - pulse_amp: Amplitude of the Gaussian pulse (optional)
    - pulse_width: Width of the Gaussian pulse (optional, small for delta-like effect)

    Returns:
    - delta_g: Fourier series perturbation or its derivative with optional Gaussian pulse (1D array)
    """
    delta_g = 0.0  # Initialized to zero

    if p is not None and p.size > 0:
        # Separate sine and cosine coefficients
        sin_coeffs = p[:M]
        cos_coeffs = p[M:]
        
        # Calculate frequency factor based on derivative order
        factor = (2 * jnp.pi * n / T) ** order
        
        # Calculate sine and cosine terms, adjusting for derivative order
        if order % 4 == 0:  # No change in sin/cos phase
            sin_terms = jnp.sin(2 * jnp.pi * n[:, None] * t[None, :] / T)
            cos_terms = jnp.cos(2 * jnp.pi * n[:, None] * t[None, :] / T)
        elif order % 4 == 1:  # First derivative
            sin_terms = jnp.cos(2 * jnp.pi * n[:, None] * t[None, :] / T)
            cos_terms = -jnp.sin(2 * jnp.pi * n[:, None] * t[None, :] / T)
        elif order % 4 == 2:  # Second derivative
            sin_terms = -jnp.sin(2 * jnp.pi * n[:, None] * t[None, :] / T)
            cos_terms = -jnp.cos(2 * jnp.pi * n[:, None] * t[None, :] / T)
        else:  # Third derivative
            sin_terms = -jnp.cos(2 * jnp.pi * n[:, None] * t[None, :] / T)
            cos_terms = jnp.sin(2 * jnp.pi * n[:, None] * t[None, :] / T)

        # Apply factor for each order to coefficients and calculate delta_g
        delta_g = jnp.dot(sin_coeffs * factor, sin_terms) + jnp.dot(cos_coeffs * factor, cos_terms)
    
    # Optionally add a Gaussian pulse and its derivatives if pulse_time is specified
    if pulse_time is not None and pulse_amp != 0 and pulse_width != 0:
        # Gaussian pulse and its derivatives
        t_diff = t - pulse_time
        gaussian_base = pulse_amp / jnp.sqrt(2 * jnp.pi * pulse_width**2)
        
        if order == 0:
            # 0th derivative: Gaussian pulse itself
            gaussian_pulse = gaussian_base * jnp.exp(-t_diff**2 / (2 * pulse_width**2))
        elif order == 1:
            # 1st derivative
            gaussian_pulse = gaussian_base * (-t_diff / pulse_width**2) * jnp.exp(-t_diff**2 / (2 * pulse_width**2))
        elif order == 2:
            # 2nd derivative
            gaussian_pulse = gaussian_base * ((t_diff**2 - pulse_width**2) / pulse_width**4) * jnp.exp(-t_diff**2 / (2 * pulse_width**2))
        elif order == 3:
            # 3rd derivative
            gaussian_pulse = gaussian_base * (-t_diff * (t_diff**2 - 3 * pulse_width**2) / pulse_width**6) * jnp.exp(-t_diff**2 / (2 * pulse_width**2))
        else:
            # Higher derivatives are not implemented here
            raise ValueError("Higher derivatives are not implemented for the Gaussian pulse")

        # Add the Gaussian pulse or its derivative to delta_g
        delta_g += gaussian_pulse

    return delta_g

def calculate_f(p_opt, config: PerturbationConfig, order=0):
    """
    Generalized function to calculate f(t) and its derivatives using log-dampened approach.
    """
    # Compute delta_g_user and delta_g_opt (keep existing code)
    delta_g_user = calculate_delta_g(
        t=config._t, p=config.p_user,
        M=config.M_user, T=config.T,
        n=config.n_user, order=order,
        pulse_time=config.pulse_time,
        pulse_amp=config.pulse_amp,
        pulse_width=config.pulse_width
    )

    delta_g_opt = calculate_delta_g(
        t=config._t, p=p_opt,
        M=config.M_opt, T=config.T,
        n=config.n_opt, order=order
    )

    # Total g(t) or its derivatives
    g_t = delta_g_user + delta_g_opt

    if order == 0:
        # Use log1p(exp(x)) for numerical stability
        # This grows more slowly than exp(x) while maintaining monotonicity
        dampened_exp = jnp.log1p(jnp.exp(g_t))
        
        # Compute the integral using cumulative sum
        integral_dampened = jnp.cumsum(dampened_exp) * config.dt
        f_t = config._t + integral_dampened
        return f_t
        
    elif order == 1:
        # First derivative: f'(t) = 1 + exp(g(t))/(1 + exp(g(t)))
        # This is the derivative of log1p(exp(x))
        f_prime = 1 + jnp.exp(g_t) / (1 + jnp.exp(g_t))
        return f_prime
        
    elif order == 2:
        # Second derivative: f''(t) = exp(g(t))*g'(t)/(1 + exp(g(t)))^2
        e_gt = jnp.exp(g_t)
        denominator = (1 + e_gt) ** 2
        
        # Compute g'(t)
        delta_g_user_prime = calculate_delta_g(
            t=config._t, p=config.p_user, 
            M=config.M_user, T=config.T, 
            n=config.n_user, order=1,
            pulse_time=config.pulse_time,
            pulse_amp=config.pulse_amp,
            pulse_width=config.pulse_width
        )
        
        delta_g_opt_prime = calculate_delta_g(
            t=config._t, p=p_opt,
            M=config.M_opt, T=config.T,
            n=config.n_opt, order=1
        )
        
        g_prime = delta_g_user_prime + delta_g_opt_prime
        f_double_prime = (e_gt * g_prime) / denominator
        return f_double_prime
        
    elif order == 3:
        # Third derivative involves chain rule and product rule
        e_gt = jnp.exp(g_t)
        denominator = (1 + e_gt) ** 3
        
        # Get g'(t), g''(t), and g'''(t)
        derivatives = []
        for derivative_order in range(1, 4):
            delta_g_user_n = calculate_delta_g(
                t=config._t, p=config.p_user,
                M=config.M_user, T=config.T,
                n=config.n_user, order=derivative_order,
                pulse_time=config.pulse_time,
                pulse_amp=config.pulse_amp,
                pulse_width=config.pulse_width
            )
            
            delta_g_opt_n = calculate_delta_g(
                t=config._t, p=p_opt,
                M=config.M_opt, T=config.T,
                n=config.n_opt, order=derivative_order
            )
            
            derivatives.append(delta_g_user_n + delta_g_opt_n)
        
        g_prime, g_double_prime, g_triple_prime = derivatives
        
        # Computing f'''(t) using the chain rule
        term1 = e_gt * g_triple_prime
        term2 = e_gt * g_prime * g_double_prime
        term3 = -2 * e_gt * g_prime ** 3
        
        f_triple_prime = (term1 + term2 + term3) / denominator
        return f_triple_prime
        
    else:
        raise ValueError("Order higher than 3 is not implemented.")

def schwarzian_derivative(p_opt, config: PerturbationConfig):
    """
    Computes the Schwarzian derivative S(f, t) for the given f(t).

    Parameters:
    - p_opt: Optimizer-controlled Fourier coefficients (1D array)
    - config: PerturbationConfig object containing user and optimizer parameters

    Returns:
    - S: Schwarzian derivative (1D array)
    """
    fp = calculate_f(p_opt, config, order=1)
    fpp = calculate_f(p_opt, config, order=2)
    fppp = calculate_f(p_opt, config, order=3)
    S = fppp / fp - 1.5 * (fpp / fp) ** 2
    return S

def schwarzian_action(p_opt, config: PerturbationConfig):
    """
    Computes the Schwarzian action by integrating the Schwarzian derivative.

    Parameters:
    - p_opt: Optimizer-controlled Fourier coefficients (1D array)
    - config: PerturbationConfig object containing user and optimizer parameters

    Returns:
    - action: Schwarzian action (scalar)
    """
    S = schwarzian_derivative(p_opt, config)
    action = -config.C * jax.scipy.integrate.trapezoid(S, config._t)
    return action

def action_to_minimize(p_opt, config: PerturbationConfig):
    """
    Objective function to minimize: the Schwarzian action.

    Parameters:
    - p_opt: Optimizer-controlled Fourier coefficients (1D array)
    - config: PerturbationConfig object containing user and optimizer parameters

    Returns:
    - action: Schwarzian action (scalar)
    """
    return schwarzian_action(p_opt, config)

def run_optimizations(action_to_minimize, p_initial, config):
    """
    Executes specified optimization methods based on the configuration.

    Parameters:
    - action_to_minimize (function): The function to minimize.
    - p_initial (array): Initial parameter guess.
    - config (dict): Configuration dict specifying which optimizers to use.

    Returns:
    - dict: A dictionary with method names as keys and optimized results as values.
    """

    # Define a list of optimization methods based on the configuration
    optimization_methods = [
        ("BFGS", run_bfgs_optimization, (action_to_minimize, p_initial)) if config["BFGS"] else None,
        ("Adam (JAX)", run_adam_optimization, (action_to_minimize, p_initial)) if config["Adam (JAX)"] else None,
        ("Adam (Optax)", run_optax_adam_optimization, (action_to_minimize, p_initial)) if config["Adam (Optax)"] else None,
        ("Yogi", run_yogi_optimization, (action_to_minimize, p_initial)) if config["Yogi"] else None,
        ("LBFGS", run_lbfgs_optimization, (action_to_minimize, p_initial)) if config["LBFGS"] else None,
        ("AdaBelief", run_adabelief_optimization, (action_to_minimize, p_initial)) if config["AdaBelief"] else None,
        ("Newton's Method", run_newtons_method, (action_to_minimize, p_initial)) if config["Newton's Method"] else None,
        ("Hessian-based Optimization", run_hessian_optimization, (action_to_minimize, p_initial)) if config["Hessian-based Optimization"] else None
    ]

    # Filter out None values (disabled methods)
    optimization_methods = [method for method in optimization_methods if method is not None]

    # Initialize dictionaries to store results
    results = {
        "optimized_params": {},
        "action_values": {},
        "times_taken": {}
    }

    # Run each optimization method and collect results
    for method_name, optimization_function, args in optimization_methods:
        p_optimal, action_value, time_taken = optimization_function(*args)
        results["optimized_params"][method_name] = p_optimal
        results["action_values"][method_name] = action_value
        results["times_taken"][method_name] = time_taken

    return results

def reparameterise_ft(action_to_minimize, p_initial, config, pert_config):
    """
    Runs optimization to reparameterize f(t) and includes f(t) in the results.

    Parameters:
    - action_to_minimize (function): The function to minimize.
    - p_initial (array): Initial parameter guess.
    - config (dict): Configuration dict specifying which optimizers to use.
    - pert_config (PerturbationConfig): Perturbation configuration for calculating f(t).

    Returns:
    - dict: A dictionary with method names as keys, containing optimized results,
      including optimized parameters, action values, times taken, and f(t) values.
    """
    # Run optimizations and get base results
    results = run_optimizations(action_to_minimize, p_initial, config)

    # Calculate f(t) for each optimizer's result and add to results
    results["f_t"] = {}
    for method_name, p_optimal in results["optimized_params"].items():
        f_t_values = calculate_f(p_optimal, pert_config, order=0)
        results["f_t"][method_name] = f_t_values  # Store f(t) values for each optimizer

    return results
