# results.py

import jax.numpy as jnp
import matplotlib.pyplot as plt

from config import PerturbationConfig  # Import the configuration class

def select_best_optimizer(results):
    """
    Selects the optimizer with the action value closest to zero from the results 
    and returns a dictionary containing its data.

    Parameters:
    - results (dict): Dictionary containing action values, times taken, f(t) values, and optimized parameters.

    Returns:
    - dict: A dictionary `best_result` containing the data of the best optimizer:
      - 'method': Name of the best optimizer method.
      - 'action_value': Action value closest to zero achieved.
      - 'time_taken': Computation time taken by the best optimizer.
      - 'f_t': The f(t) array for the best optimizer.
      - 'optimized_params': The optimized parameters (coefficients) for the best optimizer.
    """
    # Identify the optimizer with the action value closest to zero
    best_method = min(results["action_values"], key=lambda k: abs(results["action_values"][k]))
    best_result = {
        "method": best_method,
        "action_value": results["action_values"][best_method],
        "time_taken": results["times_taken"][best_method],
        "f_t": results["f_t"][best_method],
        "optimized_params": results["optimized_params"][best_method]
    }
    
    # Print a simple statement about the selected optimizer
    print(f"Selected best optimizer: {best_method}.")
    
    return best_result

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
    Plot the boundary reparameterization showing how f(t) modifies the boundary.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing optimization results including f(t) values
    config : PerturbationConfig
        Configuration instance containing parameters and time grid
    """
    plt.figure(figsize=(12, 8))
    
    t = config.t  # Use time grid from config
    
    # Plot original time coordinate
    plt.plot(t, t, 'k--', label='Original (t)', alpha=0.5)
    
    # Plot f(t) for each optimization method
    for method, f_t_values in results["f_t"].items():
        if isinstance(f_t_values, jnp.ndarray) and f_t_values.shape == t.shape:
            plt.plot(t, f_t_values, label=f'f(t) using {method}')
    
    # Customize the plot
    plt.xlabel('Original time coordinate (t)')
    plt.ylabel('Reparameterized time coordinate f(t)')
    plt.title('Boundary Reparameterization for Different Optimization Methods')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add configuration parameters as text
    config_text = (f"T={config.T}, Z={config.Z}\n"
                  f"N={config.N}\n"
                  f"Perturbation strength={config.perturbation_strength}\n")
    
    plt.text(0.02, 0.98, config_text, 
             transform=plt.gca().transAxes,
             verticalalignment='top',
             fontsize=8,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.show()

def plot_deviation_from_t(results, config: PerturbationConfig):
    """
    Plot the deviation of f(t) from linearity for each optimization method using pre-calculated values.

    Parameters:
    -----------
    results : dict
        Dictionary containing optimization results including f(t) values
    config : PerturbationConfig
        Configuration instance containing parameters and time grid
    """
    plt.figure(figsize=(12, 8))
    
    t = config.t  # Time grid from config
    
    # Plot reference line at zero (no deviation)
    plt.plot(t, jnp.zeros_like(t), 'k--', label='No deviation', alpha=0.5)
    
    # Plot deviations for each optimization method
    for method, f_t_values in results["f_t"].items():
        if isinstance(f_t_values, jnp.ndarray) and f_t_values.shape == t.shape:
            f_t_minus_t = f_t_values - t
            plt.plot(t, f_t_minus_t, label=f"Deviation using {method}")
        else:
            print(f"Skipping {method} due to incompatible result shape or type.")

    plt.xlabel('Original time coordinate (t)')
    plt.ylabel('Deviation from original time (f(t) - t)')
    plt.title('Deviation of Optimized f(t) from Original Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add configuration parameters as text
    config_text = (f"T={config.T}, Z={config.Z}\n"
                  f"N={config.N}\n"
                  f"Perturbation strength={config.perturbation_strength}\n")
    
    plt.text(0.02, 0.98, config_text, 
             transform=plt.gca().transAxes,
             verticalalignment='top',
             fontsize=8,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.show()

def plot_reparameterization(results, config: PerturbationConfig):
    """
    Plot the actual shape of the boundary in (t,z) coordinates for different optimization methods.
    
    Parameters:
    -----------
    results : dict
        Dictionary containing optimization results including f(t) values
    config : PerturbationConfig
        Configuration instance containing parameters and time grid
    save_path : str, optional
        Path to save the plot. If None, display the plot instead
    """
    plt.figure(figsize=(12, 8))
    
    t = config.t 
    z = config.Z * jnp.ones_like(t)  # Constant z for the boundary
    
    # Plot original boundary
    plt.plot(t, z, 'k--', label='Original boundary', alpha=0.5)
    
    # Plot perturbed boundary for each optimization method
    for method, f_t_values in results["f_t"].items():
        if isinstance(f_t_values, jnp.ndarray) and f_t_values.shape == t.shape:
            plt.plot(f_t_values, z, label=f'Boundary using {method}')
    
    plt.xlabel('Time coordinate')
    plt.ylabel('Spatial coordinate (z)')
    plt.title('Boundary Shape for Different Optimization Methods')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add configuration parameters
    config_text = (f"T={config.T}, Z={config.Z}\n"
                  f"N={config.N}\n"
                  f"Perturbation strength={config.perturbation_strength}\n")

    
    plt.text(0.02, 0.98, config_text, 
             transform=plt.gca().transAxes,
             verticalalignment='top',
             fontsize=8,
             bbox=dict(facecolor='white', alpha=0.8))
    
    plt.show()
