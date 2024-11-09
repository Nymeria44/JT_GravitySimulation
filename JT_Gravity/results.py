# results.py

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from config import PerturbationConfig  # Import the configuration class

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



def plot_f_vs_ft(results, config: PerturbationConfig):
    """
    Plot the optimized function f(t) for each optimization method using pre-calculated values.

    Parameters:
    - results (dict): Dictionary containing optimized parameters, f(t) values, etc.
    - config (PerturbationConfig): Configuration instance containing all necessary parameters.
    """

    t = config.t  # Time grid from config
    plt.figure(figsize=(12, 8))

    for method, f_t_values in results["f_t"].items():
        if isinstance(f_t_values, jnp.ndarray) and f_t_values.shape == t.shape:
            plt.plot(t, f_t_values, label=f"Optimized f(t) using {method}")
        else:
            print(f"Skipping {method} due to incompatible result shape or type.")

    plt.xlabel("t")
    plt.ylabel("f(t)")
    plt.title("Optimized Reparameterization of f(t) for Each Method")
    plt.legend()
    plt.show()

def plot_deviation_from_f(results, config: PerturbationConfig):
    """
    Plot the deviation of f(t) from linearity for each optimization method using pre-calculated values.

    Parameters:
    - results (dict): Dictionary containing optimized parameters, f(t) values, etc.
    - config (PerturbationConfig): Configuration instance containing all necessary parameters.
    """

    t = config.t  # Time grid from config
    plt.figure(figsize=(12, 8))

    for method, f_t_values in results["f_t"].items():
        if isinstance(f_t_values, jnp.ndarray) and f_t_values.shape == t.shape:
            f_t_minus_t = f_t_values - t
            plt.plot(t, f_t_minus_t, label=f"Deviation (f(t) - t) using {method}")
        else:
            print(f"Skipping {method} due to incompatible result shape or type.")

    plt.xlabel("t")
    plt.ylabel("f(t) - t")
    plt.title("Deviation of Optimized f(t) from Linearity for Each Method")
    plt.legend()
    plt.show()
