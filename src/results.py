import jax.numpy as jnp
import matplotlib.pyplot as plt

from config import PerturbationConfig

################################################################################
# Setting up variables
################################################################################

PLOT_CONFIG = {
    'figure.figsize': (12, 8),
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'legend.fontsize': 9,
    'legend.framealpha': 0.8,
    'legend.edgecolor': 'gray',
    'legend.facecolor': 'white',
    'grid.alpha': 0.3,
    'lines.linewidth': 2,
    'lines.markersize': 6
}

# Config of reference line for plots
REFERENCE_STYLE = {
    'alpha': 0.5,
    'linestyle': '--',
    'color': 'k'
}

def setup_plot_style():
    """Set consistent style for all plots."""
    plt.rcParams.update(PLOT_CONFIG)

def add_config_info(ax, config):
    """Add configuration parameters to plot in consistent location using legend styling."""
    config_text = (f"T={config.T}, Z={config.Z}\n"
                  f"N={config.N}\n"
                  f"Perturbation strength={config.perturbation_strength}")
    
    # Get legend properties from rcParams
    legend_props = {
        'fontsize': plt.rcParams['legend.fontsize'],
        'framealpha': plt.rcParams['legend.framealpha'],
        'edgecolor': plt.rcParams['legend.edgecolor'],
        'facecolor': plt.rcParams['legend.facecolor']
    }
    
    ax.text(0.02, 0.98, config_text,
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=legend_props['fontsize'],
            bbox=dict(
                facecolor=legend_props['facecolor'],
                alpha=legend_props['framealpha'],
                edgecolor=legend_props['edgecolor']
            ))

################################################################################
# Isolating the Best Optimizer
################################################################################

def select_best_optimizer(results):
    """
    Selects the optimizer with the action value closest to zero from the results.

    Parameters
    ----------
    results : dict
        Dictionary containing optimization results including:
        - action_values
        - times_taken
        - f_t
        - optimized_params

    Returns
    -------
    dict
        Best optimizer results containing:
        - method : str
            Name of the best optimizer method
        - action_value : float
            Action value closest to zero achieved
        - time_taken : float
            Computation time taken by the best optimizer
        - f_t : ndarray
            The f(t) array for the best optimizer
        - optimized_params : ndarray
            The optimized parameters for the best optimizer
    """
    best_method = min(results["action_values"], key=lambda k: abs(results["action_values"][k]))
    best_result = {
        "method": best_method,
        "action_value": results["action_values"][best_method],
        "time_taken": results["times_taken"][best_method],
        "f_t": results["f_t"][best_method],
        "optimized_params": results["optimized_params"][best_method]
    }
    
    print(f"Selected best optimizer: {best_method}.")
    
    return best_result

################################################################################
# Displaying Results
################################################################################

def print_optimization_results(results, verbose=False):
    """
    Print optimization results summary and optionally detailed information.

    Parameters
    ----------
    results : dict
        Dictionary containing optimization results including:
        - action_values
        - times_taken
        - f_t
        - optimized_params
    verbose : bool, default=False
        Whether to print detailed information including f(t) arrays and coefficients
    """
    print("\n" + "="*60)
    print(f"{'Final Action Values and Time Comparison':^60}")
    print("="*60)
    print(f"{'Optimizer':30} | {'Action':12} | {'Time':8}")
    print("-"*60)
    
    methods = sorted(
        results["action_values"].keys(),
        key=lambda x: float('inf') if jnp.isnan(results["action_values"][x]) 
        else results["action_values"][x]
    )
    
    for method in methods:
        action = results["action_values"][method]
        time = results["times_taken"][method]
        print(f"{method:30} | {action:12.6f} | {time:8.4f}s")
    
    print("="*60)

    if verbose:
        print("\nDetailed Results:")
        print("="*80)
        for method_name in methods:
            print(f"\n{method_name}:")
            print("-"*80)
            
            f_t_values = results["f_t"].get(method_name)
            if f_t_values is not None:
                print("f(t) Summary:")
                print(f"  First 5 values: {f_t_values[:5]}")
                print(f"  Last 5 values: {f_t_values[-5:]}")
                print("  Statistics:")
                print(f"    Min: {f_t_values.min():.6f}")
                print(f"    Max: {f_t_values.max():.6f}")
                print(f"    Mean: {f_t_values.mean():.6f}")
                print(f"    Std: {f_t_values.std():.6f}")

            optimized_params = results["optimized_params"].get(method_name)
            if optimized_params is not None:
                print("\n  Optimized Parameters:")
                print(f"  {optimized_params}")
            
            print("-"*80)

def plot_f_vs_ft(results, config: PerturbationConfig):
    setup_plot_style()
    _, ax = plt.subplots()  # Using _ for unused figure object
    
    t = config.t
    
    ax.plot(t, t, 
            linestyle=REFERENCE_STYLE['linestyle'],
            color=REFERENCE_STYLE['color'],
            alpha=REFERENCE_STYLE['alpha'],
            label='Original (t)')
    
    # Plot f(t) for each optimization method
    for method, f_t_values in results["f_t"].items():
        if isinstance(f_t_values, jnp.ndarray) and f_t_values.shape == t.shape:
            ax.plot(t, f_t_values, label=f'f(t) using {method}')
    
    ax.set_xlabel('Original time (t)')
    ax.set_ylabel('Reparameterised time f(t)')
    ax.set_title('Time Coordinate Reparameterisation')
    ax.grid(True)  # No explicit alpha needed
    ax.legend()
    
    add_config_info(ax, config)
    plt.tight_layout()
    plt.show()

def plot_deviation_from_t(results, config: PerturbationConfig):
    setup_plot_style()
    _, ax = plt.subplots()
    
    t = config.t
    
    # Plot reference line at zero
    ax.plot(t, jnp.zeros_like(t),
            linestyle=REFERENCE_STYLE['linestyle'],
            color=REFERENCE_STYLE['color'],
            alpha=REFERENCE_STYLE['alpha'],
            label='No deviation')
    
    # Plot deviations for each optimization method
    for method, f_t_values in results["f_t"].items():
        if isinstance(f_t_values, jnp.ndarray) and f_t_values.shape == t.shape:
            f_t_minus_t = f_t_values - t
            ax.plot(t, f_t_minus_t, label=f"Deviation using {method}")

    ax.set_xlabel('Original time (t)')
    ax.set_ylabel('Deviation from original time (f(t) - t)')
    ax.set_title('Time Coordinate Shift')
    ax.grid(True)
    ax.legend()
    
    add_config_info(ax, config)
    plt.tight_layout()
    plt.show()

def plot_reparameterization(results, config: PerturbationConfig):
    setup_plot_style()
    _, ax = plt.subplots()
    
    t = config.t
    z = config.Z * jnp.ones_like(t)
    
    # Plot original boundary
    ax.plot(t, z,
            linestyle=REFERENCE_STYLE['linestyle'],
            color=REFERENCE_STYLE['color'],
            alpha=REFERENCE_STYLE['alpha'],
            label='Original boundary')
    
    # Plot perturbed boundary for each optimization method
    for method, f_t_values in results["f_t"].items():
        if isinstance(f_t_values, jnp.ndarray) and f_t_values.shape == t.shape:
            ax.plot(f_t_values, z, label=f'Boundary using {method}')
    
    ax.set_xlabel('Original time coordinate (t)')
    ax.set_ylabel('Spatial coordinate (z)')
    ax.set_title('Physical Movement of Boundary')
    ax.grid(True)
    ax.legend()
    
    add_config_info(ax, config)
    plt.tight_layout()
    plt.show()
