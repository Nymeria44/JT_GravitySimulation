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
    # Set up constants and initial parameters
    T = 100000.0
    N = 100000
    C = 1.0
    perturbation_strength = 60.0

    # User-controlled perturbation parameters (Fixed)
    M_user = 5
    n_user = jnp.arange(1, M_user + 1)
    key_user = jax.random.PRNGKey(1)
    p_user = jax.random.normal(key_user, shape=(2 * M_user,)) * 0.01

    # Optimizer-controlled parameters (Initial Guess)
    M_opt = 15
    n_opt = jnp.arange(1, M_opt + 1)
    key_opt = jax.random.PRNGKey(0)
    p_initial = jax.random.normal(key_opt, shape=(2 * M_opt,)) * 0.01

    t = jnp.linspace(0.001, T, N)

    # Set optimizer configuration
    config = OPTIMIZER_CONFIG

    # Define the objective function to minimize, with p_user as a constant parameter
    def objective_function(p_opt):
        return action_to_minimize(
            p_opt, p_user, t, C, M_opt, M_user, T, perturbation_strength, n_opt, n_user
        )

    # Run optimizations
    results = run_optimizations(
        action_to_minimize=objective_function,
        p_initial=p_initial,
        config=config
    )

    # Print results and plot
    print_optimization_results(results['action_values'], results['times_taken'])
    plot_f_vs_ft(
        results['optimized_params'], p_user, t, f, p_initial,
        M_opt, M_user, T, perturbation_strength, n_opt, n_user
    )
    plot_deviation_from_f(
        results['optimized_params'], p_user, t, f, p_initial,
        M_opt, M_user, T, perturbation_strength, n_opt, n_user
    )

if __name__ == "__main__":
    main()

