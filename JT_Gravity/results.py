# results.py

import jax.numpy as jnp
import matplotlib.pyplot as plt

from config import PerturbationConfig  # Import the configuration class

def print_optimization_results(results, verbose=False):
    """
    Print the final action values, computation time, and optionally detailed values for each optimization method.

    Parameters:
    - results (dict): Dictionary containing action values, times taken, f(t) values, and optimized parameters.
    - verbose (bool): Whether to print detailed information including f(t) arrays and coefficients.
    """

    print("\nFinal Action Values and Time Comparison:")
    for method_name, action_value in results["action_values"].items():
        print(f"{method_name}: {action_value} | Time Taken: {results['times_taken'][method_name]:.4f} seconds")
        
        if verbose:
            # Print f(t) summary
            f_t_values = results["f_t"].get(method_name)
            if f_t_values is not None:

                print(f"\nf(t) for {method_name}:")
                print(f"  First 5 values: {f_t_values[:5]}")
                print(f"  Last 5 values: {f_t_values[-5:]}")
                # Print summary statistics for f(t)
                print(f"  Summary: min={f_t_values.min()}, max={f_t_values.max()}, "
                      f"mean={f_t_values.mean()}, std={f_t_values.std()}")

            # Print optimised parameters (Fourier coefficients)
            optimized_params = results["optimized_params"].get(method_name)
            if optimized_params is not None:
                print(f"\nOptimized Parameters for {method_name}:\n{optimized_params}")
            print("\n" + "-"*50 + "\n")

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
