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

def calculate_delta_f(t, p, M, T, n, order=0):
    """
    Calculate the Fourier series terms and its derivatives up to a specified order.

    Parameters:
    - t: time variable
    - p: Fourier coefficients (sin and cos combined)
    - M: Number of terms in each series (sin and cos)
    - T: Period
    - n: Array of harmonic indices
    - order: The derivative order (0 for the function itself, 1 for first derivative, etc.)

    Returns:
    - delta_f: Fourier series perturbation or its derivative
    """
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
    return delta_f

# Define f(t) with optimizer and user perturbations
def f(t, p_opt, M_opt, config: PerturbationConfig):
    # Use precomputed harmonic indices for user perturbation
    n_user = config.n_user
    n_opt = jnp.arange(config.M_user + 1, config.M_user + M_opt + 1)

    # User-controlled perturbation
    delta_f_user = calculate_delta_f(t, config.p_user, config.M_user, config.T, n_user)
    
    # Optimizer-controlled perturbation
    delta_f_opt = calculate_delta_f(t, p_opt, M_opt, config.T, n_opt)

    # Total perturbation
    return t + delta_f_user + delta_f_opt

# First derivative f'(t)
def f_prime(t, p_opt, M_opt, config: PerturbationConfig):
    n_user = config.n_user
    n_opt = jnp.arange(config.M_user + 1, config.M_user + M_opt + 1)

    delta_f_prime_user = calculate_delta_f(t, config.p_user, config.M_user, config.T, n_user, order=1)
    delta_f_prime_opt = calculate_delta_f(t, p_opt, M_opt, config.T, n_opt, order=1)

    return 1 + delta_f_prime_user + delta_f_prime_opt

# Second derivative f''(t)
def f_double_prime(t, p_opt, M_opt, config: PerturbationConfig):
    n_user = config.n_user
    n_opt = jnp.arange(config.M_user + 1, config.M_user + M_opt + 1)

    delta_f_double_prime_user = calculate_delta_f(t, config.p_user, config.M_user, config.T, n_user, order=2)
    delta_f_double_prime_opt = calculate_delta_f(t, p_opt, M_opt, config.T, n_opt, order=2)

    return delta_f_double_prime_user + delta_f_double_prime_opt

# Third derivative f'''(t)
def f_triple_prime(t, p_opt, M_opt, config: PerturbationConfig):
    n_user = config.n_user
    n_opt = jnp.arange(config.M_user + 1, config.M_user + M_opt + 1)

    delta_f_triple_prime_user = calculate_delta_f(t, config.p_user, config.M_user, config.T, n_user, order=3)
    delta_f_triple_prime_opt = calculate_delta_f(t, p_opt, M_opt, config.T, n_opt, order=3)

    return delta_f_triple_prime_user + delta_f_triple_prime_opt

# Define the Schwarzian derivative
def schwarzian_derivative(t, p_opt, M_opt, config: PerturbationConfig):
    fp = f_prime(t, p_opt, M_opt, config)
    fpp = f_double_prime(t, p_opt, M_opt, config)
    fppp = f_triple_prime(t, p_opt, M_opt, config)
    S = fppp / fp - 1.5 * (fpp / fp) ** 2
    return S

# Define the Schwarzian action
def schwarzian_action(p_opt, t, M_opt, config: PerturbationConfig):
    S = schwarzian_derivative(t, p_opt, M_opt, config)
    action = -config.C * jax_trapezoid(S, t)
    return action

# Objective function to minimize (only p_opt is optimized)
def action_to_minimize(p_opt, t, M_opt, config: PerturbationConfig):
    return schwarzian_action(p_opt, t, M_opt, config)


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

def plot_f_vs_ft(optimized_params, p_user, t, f, p_initial, M_opt, M_user, T, perturbation_strength, n_opt, n_user):
    """
    Plot the optimized function f(t) for each optimization method.

    Parameters:
    - optimized_params (dict): Optimized parameters for each method.
    - p_user (array): User-controlled perturbation parameters.
    - t (array): Time grid.
    - f (function): Function to calculate f(t, p_opt, p_user).
    - p_initial (array): Initial optimizer parameter array.
    - M_opt, M_user, T, perturbation_strength, n_opt, n_user: Additional parameters for f.
    """

    expected_shape = p_initial.shape
    plt.figure(figsize=(12, 8))

    for method, p_optimal in optimized_params.items():
        if isinstance(p_optimal, jnp.ndarray) and p_optimal.shape == expected_shape:
            f_optimal = f(t, p_optimal, p_user, M_opt, M_user, T, perturbation_strength, n_opt, n_user)
            plt.plot(t, f_optimal, label=f"Optimized f(t) using {method}")
        else:
            print(f"Skipping {method} due to incompatible result shape or type.")

    plt.xlabel("t")
    plt.ylabel("f(t)")
    plt.title("Optimized Reparameterization of f(t) for Each Method")
    plt.legend()
    plt.show()

def plot_deviation_from_f(optimized_params, p_user, t, f, p_initial, M_opt, M_user, T, perturbation_strength, n_opt, n_user):
    """
    Plot the deviation of f(t) from linearity for each optimization method.

    Parameters:
    - optimized_params (dict): Optimized parameters for each method.
    - p_user (array): User-controlled perturbation parameters.
    - t (array): Time grid.
    - f (function): Function to calculate f(t, p_opt, p_user).
    - p_initial (array): Initial optimizer parameter array.
    - M_opt, M_user, T, perturbation_strength, n_opt, n_user: Additional parameters for f.
    """

    expected_shape = p_initial.shape
    plt.figure(figsize=(12, 8))

    for method, p_optimal in optimized_params.items():
        if isinstance(p_optimal, jnp.ndarray) and p_optimal.shape == expected_shape:
            f_optimal = f(t, p_optimal, p_user, M_opt, M_user, T, perturbation_strength, n_opt, n_user)
            f_t_minus_t = f_optimal - t
            plt.plot(t, f_t_minus_t, label=f"Deviation (f(t) - t) using {method}")
        else:
            print(f"Skipping {method} due to incompatible result shape or type.")

    plt.xlabel("t")
    plt.ylabel("f(t) - t")
    plt.title("Deviation of Optimized f(t) from Linearity for Each Method")
    plt.legend()
    plt.show()
