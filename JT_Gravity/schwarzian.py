# schwarzianJAX.py

import jax.numpy as jnp
import matplotlib.pyplot as plt

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


def calculate_delta_f(t, p, M, T, n, order=0, pulse_time=None, pulse_amp=0, pulse_width=0.01):
    """
    Calculate the Fourier series terms and its derivatives up to a specified order,
    and optionally add a Gaussian pulse (and its derivatives) to simulate a Dirac delta function approximation.

    Parameters:
    - t: time variable
    - p: Fourier coefficients (sin and cos combined)
    - M: Number of terms in each series (sin and cos)
    - T: Period
    - n: Array of harmonic indices
    - order: The derivative order (0 for the function itself, 1 for first derivative, etc.)
    - pulse_time: Center time for the Gaussian pulse (optional)
    - pulse_amplitude: Amplitude of the Gaussian pulse (optional)
    - pulse_width: Width of the Gaussian pulse (optional, small for delta-like effect)

    Returns:
    - delta_f: Fourier series perturbation or its derivative with optional Gaussian pulse
    """
    delta_f = 0  # Initialised to zero

    if p is not None:
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

        # Apply factor for each order to coefficients and calculate delta_f
        delta_f = jnp.dot(sin_coeffs * factor, sin_terms) + jnp.dot(cos_coeffs * factor, cos_terms)
    
    # Optionally add a Gaussian pulse and its derivatives if pulse_time is specified
    if pulse_time is not None:
        # Gaussian pulse and its derivatives
        t_diff = t - pulse_time
        gaussian_base = pulse_amp * (1 / jnp.sqrt(2 * jnp.pi * pulse_width**2))
        
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

        # Add the Gaussian pulse or its derivative to delta_f
        delta_f += gaussian_pulse
    
    return delta_f

def calculate_f(p_opt, config: PerturbationConfig, order=0):
    """
    Generalized function to calculate f(t) and its derivatives up to the specified order.
    
    Parameters:
    - p_opt: Optimizer-controlled Fourier coefficients
    - config: PerturbationConfig object containing user and optimizer parameters
    - order: The derivative order (0 for f(t), 1 for f'(t), etc.)

    Returns:
    - f_derivative: The specified derivative of f(t)
    """
    t = config.t


    # User perturbation derivative of the specified order
    delta_f_user = calculate_delta_f(
        t, config.p_user, config.M_user,
        config.T, config.n_user,
        order=order,
        pulse_time=config.pulse_time,
        pulse_amp=config.pulse_amp,
        pulse_width=config.pulse_width
    )

    # Optimizer perturbation derivative of the specified order
    delta_f_opt = calculate_delta_f(
        t, p_opt, config.M_opt,
        config.T, config.n_opt,
        order=order
    )

    # Combine user and optimizer perturbations with baseline
    if order == 0:
        return t + delta_f_user + delta_f_opt
    else:
        return delta_f_user + delta_f_opt + (1 if order == 1 else 0)


# Define the Schwarzian derivative
def schwarzian_derivative(p_opt, config: PerturbationConfig):
    fp = calculate_f(p_opt, config, 1)
    fpp = calculate_f(p_opt, config, 2)
    fppp = calculate_f(p_opt, config, 3)
    S = fppp / fp - 1.5 * (fpp / fp) ** 2
    return S

# Trapezoidal integration
def jax_trapz(y, x):
    dx = jnp.diff(x)
    return jnp.sum((y[:-1] + y[1:]) * dx / 2.0)

# Define the Schwarzian action
def schwarzian_action(p_opt, config: PerturbationConfig):
    S = schwarzian_derivative(p_opt, config)
    action = -config.C * jax_trapz(S, config.t)
    return action

# Objective function to minimize (only p_opt is optimized)
def action_to_minimize(p_opt, config: PerturbationConfig):
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

def print_optimization_results(action_values, times_taken):
    """
    Print the final action values and computation time for each optimization method.

    Parameters:
    - action_values (dict): Final action values from each optimizer.
    - times_taken (dict): Computation time taken for each optimizer.
    """

    print("\nFinal Action Values and Time Comparison:")
    for method_name, action_value in action_values.items():
        print(f"{method_name}: {action_value} | Time Taken: {times_taken[method_name]:.4f} seconds")


def plot_f_vs_ft(optimized_params, p_initial, config: PerturbationConfig):
    """
    Plot the optimized function f(t) for each optimization method.

    Parameters:
    - optimized_params (dict): Optimized parameters for each method.
    - f (function): Function to calculate f(t, p_opt, using config).
    - p_initial (array): Initial optimizer parameter array.
    - config (PerturbationConfig): Configuration instance containing all necessary parameters.
    """

    expected_shape = p_initial.shape
    t = config.t  # Time grid from config
    plt.figure(figsize=(12, 8))

    for method, p_optimal in optimized_params.items():
        if isinstance(p_optimal, jnp.ndarray) and p_optimal.shape == expected_shape:
            # Calculate f_optimal using config and optimized parameters
            f_optimal = calculate_f(p_optimal, config, 0)
            plt.plot(t, f_optimal, label=f"Optimized f(t) using {method}")
        else:
            print(f"Skipping {method} due to incompatible result shape or type.")

    plt.xlabel("t")
    plt.ylabel("f(t)")
    plt.title("Optimized Reparameterization of f(t) for Each Method")
    plt.legend()
    plt.show()

def plot_deviation_from_f(optimized_params, p_initial, config: PerturbationConfig):
    """
    Plot the deviation of f(t) from linearity for each optimization method.

    Parameters:
    - optimized_params (dict): Optimized parameters for each method.
    - f (function): Function to calculate f(t, p_opt, using config).
    - p_initial (array): Initial optimizer parameter array.
    - config (PerturbationConfig): Configuration instance containing all necessary parameters.
    """

    expected_shape = p_initial.shape
    t = config.t  # Time grid from config
    plt.figure(figsize=(12, 8))

    for method, p_optimal in optimized_params.items():
        if isinstance(p_optimal, jnp.ndarray) and p_optimal.shape == expected_shape:
            # Calculate f_optimal using config and optimized parameters
            f_optimal = calculate_f(p_optimal, config, 0)
            f_t_minus_t = f_optimal - t
            plt.plot(t, f_t_minus_t, label=f"Deviation (f(t) - t) using {method}")
        else:
            print(f"Skipping {method} due to incompatible result shape or type.")

    plt.xlabel("t")
    plt.ylabel("f(t) - t")
    plt.title("Deviation of Optimized f(t) from Linearity for Each Method")
    plt.legend()
    plt.show()
