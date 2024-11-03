# __init__.py file for JT_Gravity package
import jax
import jax.numpy as jnp
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

from config import PerturbationConfig

# Environment setup
os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'
matplotlib.use('TkAgg')

# Configuration dictionary to enable/disable specific optimizers
OPTIMIZER_CONFIG = {
    "BFGS": True,
    "Adam (JAX)": True,
    "Adam (Optax)": True,
    "Yogi": True,
    "LBFGS": True,
    "AdaBelief": True,
    "Newton's Method": False,
    "Hessian-based Optimization": True
}

def main():
    # Initialize perturbation configuration, including user and optimizer parameters
    config = PerturbationConfig(
        T=100000.0,
        N=100000,
        C=1.0,
        perturbation_strength=60.0,
        M_user=5,
        M_opt=15
    )

    # Initial guess for optimizer-controlled parameters
    key_opt = jax.random.PRNGKey(0)
    p_initial = jax.random.normal(key_opt, shape=(2 * config.M_opt,)) * 0.01

    # Define the objective function to minimize, with config encapsulating all parameters
    def objective_function(p_opt):
        return action_to_minimize(p_opt, config)

    # Run optimizations
    results = run_optimizations(
        action_to_minimize=objective_function,
        p_initial=p_initial,
        config=OPTIMIZER_CONFIG
    )

    # Print results and plot
    print_optimization_results(results['action_values'], results['times_taken'])
    plot_f_vs_ft(results['optimized_params'], f, p_initial, config)
    plot_deviation_from_f(results['optimized_params'], f, p_initial, config)

if __name__ == "__main__":
    main()
