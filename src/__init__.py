# __init__.py

import jax
import matplotlib
import os
import sys  # To check command-line arguments

from schwarzian import (
    action_to_minimize,
    reparameterise_ft
)

from results import (
    print_optimization_results,
    plot_f_vs_ft,
    plot_deviation_from_f,
    select_best_optimizer
)

from config import PerturbationConfig
from harmonic_sweep import harmonic_sweep  # Import the sweep function

os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'
matplotlib.use('TkAgg')

OPTIMIZER_CONFIG = {
    "BFGS": True,
    "Adam (JAX)": True,
    "Adam (Optax)": True,
    "Yogi": True,
    "LBFGS": True,
    "AdaBelief": True,
    "Newton's Method": True,
    "Hessian-based Optimization": True 
}

def main():
    PertConfig = PerturbationConfig(
        # Core parameters
        T=10.0,                   # Total sim time
        Z = 10.0,
        N=10000,                  # Number of time samples
        G = 1,  # Gravitational constant in 2D
        a = 1,  # back reaction stability parameter (postive constant)

        # Fourier perturbation settings
        perturbation_strength=1, # Magnitude of user Fourier Pertubation
        M_user=5,                 # Number of Fourier series harmonics (split 50/50 between user and optimizer)
        M_opt=20,

        # Gaussian pulse settings
        pulse_time=0,             # Center of Gaussian pulse
        pulse_amp=0,              # Amplitude of Gaussian pulse
        pulse_width=0             # Width of Gaussian pulse
    )
    PertConfig.validate_pulse_width()
    PertConfig.debug_info()

    # Initial guess for optimizer-controlled parameters
    key_opt = jax.random.PRNGKey(0)
    p_initial = jax.random.normal(key_opt, shape=(2 * PertConfig.M_opt,)) * 0.0001

    # Define the objective function to minimize, with config encapsulating all parameters
    def objective_function(p):
        return action_to_minimize(p, PertConfig)

    # Run optimizations
    results = reparameterise_ft(
        action_to_minimize=objective_function,
        p_initial=p_initial,
        config=OPTIMIZER_CONFIG,
        pert_config=PertConfig
    )

    # Print results and plot for optimizers
    print_optimization_results(results, verbose=False)
    plot_f_vs_ft(results, PertConfig)
    plot_deviation_from_f(results, PertConfig)

    f_t = select_best_optimizer(results)

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        harmonic_sweep()
    else:
        main()