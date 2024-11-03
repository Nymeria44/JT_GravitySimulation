# __init__.py file for JT_Gravity package
import jax
import jax.numpy as jnp
from jax import grad
import matplotlib
import os

from schwarzianJAX import (
    f,
    run_optimizations,
    action_to_minimize,
    print_optimization_results,
    plot_f_vs_ft,
    plot_deviation_from_f
)

# Environment setup
os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'
matplotlib.use('TkAgg')

# Configuration dictionary to enable/disable specific optimizers
OPTIMIZER_CONFIG = {
    "BFGS": False,
    "Adam (JAX)": True,
    "Adam (Optax)": True,
    "Yogi": True,
    "LBFGS": True,
    "AdaBelief": True,
    "Newton's Method": False,
    "Hessian-based Optimization": False 
}

def main():
    # Set up constants and initial parameters
    T, N, C, perturbation_strength, M = 100.0, 100, 1.0, 100, 40
    t = jnp.linspace(0.001, T, N)
    n = jnp.arange(1, M + 1)

    key = jax.random.PRNGKey(0)
    p_initial = jax.random.normal(key, shape=(2 * M,)) * 0.01

    # Set optimizer configuration
    config = OPTIMIZER_CONFIG

    # Run optimizations
    results = run_optimizations(
        action_to_minimize=lambda p: action_to_minimize(p, t, C, M, T, perturbation_strength, n),
        p_initial=p_initial,
        config=config
    )

    # Print results and plot
    print_optimization_results(results['action_values'], results['times_taken'])
    plot_f_vs_ft(results['optimized_params'], t, f, p_initial, M, T, perturbation_strength, n)
    plot_deviation_from_f(results['optimized_params'], t, f, p_initial, M, T, perturbation_strength, n)

if __name__ == "__main__":
    main()
