import matplotlib
import jax.numpy as jnp
import matplotlib.pyplot as plt

from config import PerturbationConfig
from ft_config import FtOptimalConfig

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.preamble": r"""
        \usepackage[utf8x]{inputenc}
        \usepackage{amsmath}
        \usepackage{amsfonts}
        \usepackage{amssymb}
        \usepackage{siunitx}
    """,
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'figure.dpi': 100,
    'savefig.dpi': 100,
})

################################################################################
# Setting up variables
################################################################################

PLOT_CONFIG = {
    'figure.figsize': (6, 4),
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
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
# Boundary Results
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
        print(f"{method:30} | {action:12.2e} | {time:8.4f}s")
    
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

def plot_f_vs_ft(results, config: PerturbationConfig, filename="f_vs_ft"):
    """
    Plot the boundary reparameterization showing how f(t) modifies the boundary.

    Parameters
    ----------
    results : dict
        Dictionary containing optimization results including f(t) values
    config : PerturbationConfig
        Configuration instance containing parameters and time grid
    """
    setup_plot_style()
    _, ax = plt.subplots()
    
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
    ax.grid(True)
    ax.legend()
    
    add_config_info(ax, config)
    plt.tight_layout()
    plt.show()

def plot_deviation_from_t(results, config: PerturbationConfig, filename="deviation_from_t"):
    """
    Plot the deviation of f(t) from linearity.

    Parameters
    ----------
    results : dict
        Dictionary containing optimization results including f(t) values
    config : PerturbationConfig
        Configuration instance containing parameters and time grid
    """
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
    # plt.show()

    plt.savefig(f"{filename}.pgf", bbox_inches='tight', pad_inches=0.1)
    plt.close()

def plot_boundary(results, config: PerturbationConfig, filename="boundary"):
    """
    Plot the physical boundary in (t,z) coordinates.
    The boundary lives at fixed z, and f(t) gives its time reparameterization.

    Parameters
    ----------
    results : dict
        Dictionary containing optimization results including f(t) values
    config : PerturbationConfig
        Configuration instance containing parameters and time grid
    """
    setup_plot_style()
    _, ax = plt.subplots()
    
    t = config.t
    z0 = 0  # Fixed spatial position of the boundary
    
    # Plot original boundary - a straight line at z=z0
    ax.axhline(y=z0,
               linestyle=REFERENCE_STYLE['linestyle'],
               color=REFERENCE_STYLE['color'],
               alpha=REFERENCE_STYLE['alpha'],
               label='Original boundary')
    
    # Plot reparameterized boundary for each optimization method
    for method, f_t_values in results["f_t"].items():
        if isinstance(f_t_values, jnp.ndarray) and f_t_values.shape == t.shape:
            # The boundary points move from (t,z0) to (f(t),z0)
            ax.plot(t, z0 + (f_t_values - t), label=f'Boundary using {method}')
    
    ax.set_xlabel('Time coordinate (t)')
    ax.set_ylabel('Spatial coordinate (z)')
    ax.set_title('Physical Boundary')
    ax.grid(True)
    ax.legend()
    
    add_config_info(ax, config)
    plt.tight_layout()
    # plt.show()

    plt.savefig(f"{filename}.pgf", bbox_inches='tight', pad_inches=0.1)
    plt.close()

################################################################################
# Bulk Results
################################################################################

def plot_dilaton_field(ft_config: FtOptimalConfig, pert_config: PerturbationConfig, filename="dilaton_field"):
    """
    Plot the dilaton field Φ(u,v) in light cone coordinates.

    Parameters
    ----------
    ft_config : FtConfig
        Configuration instance containing optimized f(t) and derived fields
    pert_config : PerturbationConfig
        Configuration instance containing coordinate grids and parameters
    """
    setup_plot_style()
    fig, ax = plt.subplots()
    
    # Get the dilaton field and use existing meshgrid from pert_config
    dilaton = ft_config.dilaton
    
    # Create contour plot using the existing u,v grids
    contour = ax.contourf(pert_config._u, pert_config._v, dilaton, 
                         levels=20, cmap='viridis')
    fig.colorbar(contour, ax=ax, label=r'Dilaton field $\Phi(u,v)$')
    
    # Plot the AdS boundary (u=v line)
    u_vals = jnp.linspace(pert_config.t[0], pert_config.t[-1], 100)
    ax.plot(u_vals, u_vals,
            linestyle=REFERENCE_STYLE['linestyle'],
            color=REFERENCE_STYLE['color'],
            alpha=REFERENCE_STYLE['alpha'],
            label='AdS boundary (u=v)')
    
    # Set equal aspect ratio for light cone coordinates
    ax.set_aspect('equal')
    
    ax.set_xlabel('Light cone coordinate (u = t + z)')
    ax.set_ylabel('Light cone coordinate (v = t - z)')
    ax.set_title('Dilaton Field in Light Cone Coordinates')
    ax.grid(True)
    ax.legend()
    
    add_config_info(ax, pert_config)
    plt.tight_layout()
    # plt.show()

    plt.savefig(f"{filename}.pgf", bbox_inches='tight', pad_inches=0.1)
    plt.close()
