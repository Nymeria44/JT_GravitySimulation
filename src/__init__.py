# __init__.py

import jax
import matplotlib
import os
import sys

from schwarzian import (
    action_to_minimize,
    reparameterise_ft
)

from results import (
    select_best_optimizer,
    print_optimization_results,
    plot_f_vs_ft,
    plot_deviation_from_t,
    plot_boundary
)

from harmonic_sweep import harmonic_sweep

from config import PerturbationConfig

################################################################################
# Setting up variables
################################################################################

os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'
matplotlib.use('TkAgg')

OPTIMIZER_CONFIG = {
    "BFGS": False,
    "Adam (JAX)": True,
    "Adam (Optax)": True,
    "Yogi": False,
    "LBFGS": False,
    "AdaBelief": True,
    "Newton's Method": False,
    "Hessian-based Optimization": False
}

def main():
    PertConfig = PerturbationConfig(
        # Core parameters
        T=100.0,                   # Total sim time
        Z = 10.0,
        N=15000,                  # Number of time samples
        G = 1,  # Gravitational constant in 2D
        a = 10,  # back reaction stability parameter (postive constant)

        # Fourier perturbation settings
        perturbation_strength=0.1, # Magnitude of user Fourier Pertubation
        M_user=8,                 # Number of user Fourier series harmonics
        M_opt=20,                 # Number of optimiser Fourier series harmonics

        # Gaussian pulse settings
        pulse_time=0,             # Center of Gaussian pulse
        pulse_amp=0,              # Amplitude of Gaussian pulse
        pulse_width=0             # Width of Gaussian pulse
    )
    PertConfig.validate_pulse_width()
    PertConfig.debug_info()

################################################################################
# Boundary Calculations
################################################################################

    # Initial guess for optimizer-controlled parameters
    key_opt = jax.random.PRNGKey(0)
    p_initial = jax.random.normal(key_opt, shape=(2 * PertConfig.M_opt,))

    # Define the objective function to minimize, with config encapsulating all parameters
    def objective_function(p):
        return action_to_minimize(p, PertConfig)

    # Run optimizations
    results = reparameterise_ft(
        action_to_minimize=objective_function,
        p_initial=p_initial,
        config=OPTIMIZER_CONFIG,
        pert_config=PertConfig,
        verbose=False
    )

################################################################################
# Bulk Calculations
################################################################################

    f_t = select_best_optimizer(results)

################################################################################
# Plotting and Results
################################################################################

    print_optimization_results(results, verbose=False)
    plot_f_vs_ft(results, PertConfig)
    # plot_deviation_from_t(results, PertConfig)
    plot_boundary(results, PertConfig)

################################################################################
# Main Function
################################################################################

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        harmonic_sweep()
    else:
        main()
