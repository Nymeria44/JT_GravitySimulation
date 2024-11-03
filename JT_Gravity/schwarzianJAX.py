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

# Define f(t, p_opt, p_user) with both optimizer-controlled and user-controlled perturbations
def f(t, p_opt, p_user, M_opt, M_user, T, perturbation_strength, n_opt, n_user):
    # Optimizer-controlled perturbation
    sin_coeffs_opt = p_opt[:M_opt] 
    cos_coeffs_opt = p_opt[M_opt:]
    sin_terms_opt = jnp.sin(2 * jnp.pi * n_opt[:, None] * t[None, :] / T)
    cos_terms_opt = jnp.cos(2 * jnp.pi * n_opt[:, None] * t[None, :] / T)
    delta_f_opt = jnp.dot(sin_coeffs_opt, sin_terms_opt) + jnp.dot(cos_coeffs_opt, cos_terms_opt)

    # User-controlled perturbation
    sin_coeffs_user = p_user[:M_user] * perturbation_strength
    cos_coeffs_user = p_user[M_user:] * perturbation_strength
    sin_terms_user = jnp.sin(2 * jnp.pi * n_user[:, None] * t[None, :] / T)
    cos_terms_user = jnp.cos(2 * jnp.pi * n_user[:, None] * t[None, :] / T)
    delta_f_user = jnp.dot(sin_coeffs_user, sin_terms_user) + jnp.dot(cos_coeffs_user, cos_terms_user)

    # Total perturbation
    delta_f = delta_f_opt + delta_f_user
    return t + delta_f

# First derivative f'(t)
def f_prime(t, p_opt, p_user, M_opt, M_user, T, perturbation_strength, n_opt, n_user):
    # Optimizer-controlled perturbation derivatives
    sin_coeffs_opt = p_opt[:M_opt]
    cos_coeffs_opt = p_opt[M_opt:]
    sin_deriv_opt = jnp.cos(2 * jnp.pi * n_opt[:, None] * t[None, :] / T)
    cos_deriv_opt = -jnp.sin(2 * jnp.pi * n_opt[:, None] * t[None, :] / T)
    delta_f_prime_opt = jnp.dot(
        sin_coeffs_opt * (2 * jnp.pi * n_opt / T), sin_deriv_opt
    ) + jnp.dot(
        cos_coeffs_opt * (2 * jnp.pi * n_opt / T), cos_deriv_opt
    )

    # User-controlled perturbation derivatives
    sin_coeffs_user = p_user[:M_user] * perturbation_strength
    cos_coeffs_user = p_user[M_user:] * perturbation_strength
    sin_deriv_user = jnp.cos(2 * jnp.pi * n_user[:, None] * t[None, :] / T)
    cos_deriv_user = -jnp.sin(2 * jnp.pi * n_user[:, None] * t[None, :] / T)
    delta_f_prime_user = jnp.dot(
        sin_coeffs_user * (2 * jnp.pi * n_user / T), sin_deriv_user
    ) + jnp.dot(
        cos_coeffs_user * (2 * jnp.pi * n_user / T), cos_deriv_user
    )

    # Total derivative
    delta_f_prime = delta_f_prime_opt + delta_f_prime_user
    return 1 + delta_f_prime

# Second derivative f''(t)
def f_double_prime(t, p_opt, p_user, M_opt, M_user, T, perturbation_strength, n_opt, n_user):
    # Optimizer-controlled perturbation second derivatives
    sin_coeffs_opt = p_opt[:M_opt]
    cos_coeffs_opt = p_opt[M_opt:]
    sin_double_deriv_opt = -jnp.sin(2 * jnp.pi * n_opt[:, None] * t[None, :] / T)
    cos_double_deriv_opt = -jnp.cos(2 * jnp.pi * n_opt[:, None] * t[None, :] / T)
    delta_f_double_prime_opt = jnp.dot(
        sin_coeffs_opt * ((2 * jnp.pi * n_opt / T) ** 2), sin_double_deriv_opt
    ) + jnp.dot(
        cos_coeffs_opt * ((2 * jnp.pi * n_opt / T) ** 2), cos_double_deriv_opt
    )

    # User-controlled perturbation second derivatives
    sin_coeffs_user = p_user[:M_user] * perturbation_strength
    cos_coeffs_user = p_user[M_user:] * perturbation_strength
    sin_double_deriv_user = -jnp.sin(2 * jnp.pi * n_user[:, None] * t[None, :] / T)
    cos_double_deriv_user = -jnp.cos(2 * jnp.pi * n_user[:, None] * t[None, :] / T)
    delta_f_double_prime_user = jnp.dot(
        sin_coeffs_user * ((2 * jnp.pi * n_user / T) ** 2), sin_double_deriv_user
    ) + jnp.dot(
        cos_coeffs_user * ((2 * jnp.pi * n_user / T) ** 2), cos_double_deriv_user
    )

    # Total second derivative
    delta_f_double_prime = delta_f_double_prime_opt + delta_f_double_prime_user
    return delta_f_double_prime

# Third derivative f'''(t)
def f_triple_prime(t, p_opt, p_user, M_opt, M_user, T, perturbation_strength, n_opt, n_user):
    # Optimizer-controlled perturbation third derivatives
    sin_coeffs_opt = p_opt[:M_opt]
    cos_coeffs_opt = p_opt[M_opt:]
    sin_triple_deriv_opt = -jnp.cos(2 * jnp.pi * n_opt[:, None] * t[None, :] / T)
    cos_triple_deriv_opt = jnp.sin(2 * jnp.pi * n_opt[:, None] * t[None, :] / T)
    delta_f_triple_prime_opt = jnp.dot(
        sin_coeffs_opt * ((2 * jnp.pi * n_opt / T) ** 3), sin_triple_deriv_opt
    ) + jnp.dot(
        cos_coeffs_opt * ((2 * jnp.pi * n_opt / T) ** 3), cos_triple_deriv_opt
    )

    # User-controlled perturbation third derivatives
    sin_coeffs_user = p_user[:M_user] * perturbation_strength
    cos_coeffs_user = p_user[M_user:] * perturbation_strength
    sin_triple_deriv_user = -jnp.cos(2 * jnp.pi * n_user[:, None] * t[None, :] / T)
    cos_triple_deriv_user = jnp.sin(2 * jnp.pi * n_user[:, None] * t[None, :] / T)
    delta_f_triple_prime_user = jnp.dot(
        sin_coeffs_user * ((2 * jnp.pi * n_user / T) ** 3), sin_triple_deriv_user
    ) + jnp.dot(
        cos_coeffs_user * ((2 * jnp.pi * n_user / T) ** 3), cos_triple_deriv_user
    )

    # Total third derivative
    delta_f_triple_prime = delta_f_triple_prime_opt + delta_f_triple_prime_user
    return delta_f_triple_prime

# Define the Schwarzian derivative
def schwarzian_derivative(t, p_opt, p_user, M_opt, M_user, T, perturbation_strength, n_opt, n_user):
    fp = f_prime(t, p_opt, p_user, M_opt, M_user, T, perturbation_strength, n_opt, n_user)
    fpp = f_double_prime(t, p_opt, p_user, M_opt, M_user, T, perturbation_strength, n_opt, n_user)
    fppp = f_triple_prime(t, p_opt, p_user, M_opt, M_user, T, perturbation_strength, n_opt, n_user)
    S = fppp / fp - 1.5 * (fpp / fp) ** 2
    return S

# Trapezoidal integration
def jax_trapz(y, x):
    dx = jnp.diff(x)
    return jnp.sum((y[:-1] + y[1:]) * dx / 2.0)

# Define the Schwarzian action
def schwarzian_action(p_opt, p_user, t, C, M_opt, M_user, T, perturbation_strength, n_opt, n_user):
    S = schwarzian_derivative(t, p_opt, p_user, M_opt, M_user, T, perturbation_strength, n_opt, n_user)
    action = -C * jax_trapz(S, t)
    return action

# Objective function to minimize (only p_opt is optimized)
def action_to_minimize(p_opt, p_user, t, C, M_opt, M_user, T, perturbation_strength, n_opt, n_user):
    return schwarzian_action(p_opt, p_user, t, C, M_opt, M_user, T, perturbation_strength, n_opt, n_user)

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
