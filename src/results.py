import matplotlib
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

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
})

################################################################################
# Setting up variables
################################################################################

PLOT_CONFIG = {
    'figure.figsize': (6, 4),
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.fontsize': 8,
    'legend.framealpha': 0.8,
    'legend.edgecolor': 'gray',
    'legend.facecolor': 'white',
    'grid.alpha': 0.3,
    'lines.linewidth': 1.5,
    'lines.markersize': 3
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

def add_config_info(ax, config, x=0.02, y=0.98, va='top'):
    """Add configuration parameters to plot in consistent location using legend styling.
    
    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add the text to
    config : PerturbationConfig
        Configuration instance containing parameters
    x : float, optional
        x position in axes coordinates (default: 0.02)
    y : float, optional
        y position in axes coordinates (default: 0.98)
    va : str, optional
        vertical alignment ('top' or 'bottom', default: 'top')
    """
    config_text = (r"$N=%d$, $\mathrm{Perturbation}=%.2f$" "\n"
                  r"$M_\mathrm{opt}=%d$, $M_\mathrm{user}=%d$" % 
                  (config.N, config.perturbation_strength, 
                   int(config.M_opt), int(config.M_user)))
    
    legend_props = {
        'fontsize': plt.rcParams['legend.fontsize'],
        'framealpha': plt.rcParams['legend.framealpha'],
        'edgecolor': plt.rcParams['legend.edgecolor'],
        'facecolor': plt.rcParams['legend.facecolor']
    }
    
    ax.text(x, y, config_text,
            transform=ax.transAxes,
            verticalalignment=va,
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

def downsample_array(arr, target_size=200):
    """
    Downsample a 1D or 2D array to approximately target_size points.
    
    Parameters
    ----------
    arr : array_like
        Input array (1D or 2D)
    target_size : int, optional
        Target number of points (default: 200)
        
    Returns
    -------
    array_like
        Downsampled array
    """
    arr = jnp.array(arr)  # Convert JAX arrays to numpy if needed
    if arr.ndim == 1:
        step = max(len(arr) // target_size, 1)
        return arr[::step]
    else:
        scale = target_size / max(arr.shape)
        return zoom(arr, (scale, scale))

def plot_f_vs_ft(results, config: PerturbationConfig, filename="f_vs_ft"):
    """
    Plot the boundary reparameterization showing how f(t) modifies the boundary.

    Parameters
    ----------
    results : dict
        Dictionary containing optimization results including f(t) values
    config : PerturbationConfig
        Configuration instance containing parameters and time grid
    filename : str, optional
        Output filename (default: "f_vs_ft")
    """
    setup_plot_style()
    _, ax = plt.subplots()
    
    t = downsample_array(config.t)
    
    ax.plot(t, t, 
            linestyle=REFERENCE_STYLE['linestyle'],
            color=REFERENCE_STYLE['color'],
            alpha=REFERENCE_STYLE['alpha'],
            label=r'$t$')
    
    for method, f_t_values in results["f_t"].items():
        if isinstance(f_t_values, jnp.ndarray) and f_t_values.shape == config.t.shape:
            f_t_downsampled = downsample_array(f_t_values)
            ax.plot(t, f_t_downsampled, label=rf'$f(t)$ using {method}')
    
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$f(t)$')
    ax.set_title(r'Time Coordinate Reparametrisation')
    ax.grid(True)
    ax.legend()
    
    add_config_info(ax, config)
    plt.tight_layout()
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
    filename : str, optional
        Output filename (default: "boundary")
    """
    setup_plot_style()
    _, ax = plt.subplots()
    
    t = downsample_array(config.t)
    z0 = 0
    
    ax.axhline(y=z0,
               linestyle=REFERENCE_STYLE['linestyle'],
               color=REFERENCE_STYLE['color'],
               alpha=REFERENCE_STYLE['alpha'],
               label=r'Original boundary')
    
    for method, f_t_values in results["f_t"].items():
        if isinstance(f_t_values, jnp.ndarray) and f_t_values.shape == config.t.shape:
            f_t_downsampled = downsample_array(f_t_values)
            ax.plot(t, z0 + (f_t_downsampled - t), 
                   label=rf'Boundary using {method}')
    
    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$z$')
    ax.set_title(r'Physical Boundary')
    ax.grid(True)
    ax.legend()
    
    add_config_info(ax, config)
    plt.tight_layout()
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
    filename : str, optional
        Output filename (default: "dilaton_field")
    """
    setup_plot_style()
    fig, ax = plt.subplots()
    
    dilaton_downsampled = downsample_array(ft_config.dilaton)
    u_downsampled = downsample_array(pert_config._u)
    v_downsampled = downsample_array(pert_config._v)
    
    contour = ax.contourf(u_downsampled, v_downsampled, dilaton_downsampled, 
                         levels=15, cmap='viridis')
    fig.colorbar(contour, ax=ax, label=r'$\Phi(u,v)$')
    
    u_vals = jnp.linspace(pert_config.t[0], pert_config.t[-1], 50)
    ax.plot(u_vals, u_vals,
            linestyle=REFERENCE_STYLE['linestyle'],
            color=REFERENCE_STYLE['color'],
            alpha=REFERENCE_STYLE['alpha'],
            label=r'AdS boundary ($u=v$)')
    
    ax.set_aspect('equal')
    ax.set_xlabel(r'$u = t + z$')
    ax.set_ylabel(r'$v = t - z$')
    ax.set_title(r'Dilaton Field in Light Cone Coordinates')
    ax.grid(True)
    ax.legend()
    
    add_config_info(ax, pert_config, x=0.02, y=0.02, va='bottom')
    plt.tight_layout()
    plt.savefig(f"{filename}.pgf", bbox_inches='tight', pad_inches=0.1)
    plt.close()
