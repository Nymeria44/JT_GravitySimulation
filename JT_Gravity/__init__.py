# __init__.py

import jax
import matplotlib
import os
import sys  # To check command-line arguments

from schwarzian import (
    run_optimizations,
    action_to_minimize,
    print_optimization_results,
    plot_f_vs_ft,
    plot_deviation_from_f
)
from config import PerturbationConfig
from harmonic_sweep import harmonic_sweep  # Import the sweep function

os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'
matplotlib.use('TkAgg')

OPTIMIZER_CONFIG = {
    "BFGS": False,
    "Adam (JAX)": True,
    "Adam (Optax)": True,
    "Yogi": True,
    "LBFGS": False,
    "AdaBelief": True,
    "Newton's Method": False,
    "Hessian-based Optimization": False
}

def main():
    PertConfig = PerturbationConfig(
        # Core parameters
        T=10.0,                   # Total sim time
        N=1000000,                  # Number of time samples
        G = 1,  # Gravitational constant in 2D
        a = 1,  # back reaction stability parameter (postive constant)

        # Fourier perturbation settings
        perturbation_strength=0.1, # Magnitude of user Fourier Pertubation
        M_user=5,                 # Number of Fourier series harmonics (split 50/50 between user and optimizer)
        M_opt=30,

        # Gaussian pulse settings
        pulse_time=0,             # Center of Gaussian pulse
        pulse_amp=0,              # Amplitude of Gaussian pulse
        pulse_width=0             # Width of Gaussian pulse
    )
    PertConfig.validate_pulse_width()
    PertConfig.debug_info()

    # Initial guess for optimizer-controlled parameters
    key_opt = jax.random.PRNGKey(0)
    p_initial = jax.random.normal(key_opt, shape=(2 * PertConfig.M_opt,))

    # Define the objective function to minimize, with config encapsulating all parameters
    def objective_function(p):
        return action_to_minimize(p, PertConfig)

    # Run optimizations
    results = run_optimizations(
        action_to_minimize=objective_function,
        p_initial=p_initial,
        config=OPTIMIZER_CONFIG
    )

    # Print results and plot
    print_optimization_results(results['action_values'], results['times_taken'])
    plot_f_vs_ft(results['optimized_params'], p_initial, PertConfig )
    plot_deviation_from_f(results['optimized_params'], p_initial, PertConfig)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        harmonic_sweep()
    else:
        main()
