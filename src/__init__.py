# __init__.py
import os
import sys

import jax
import matplotlib

from schwarzian import (
    action_to_minimize,
    reparameterise_ft
)

from results import (
    print_optimization_results,
    plot_f_vs_ft,
    plot_boundary,
    plot_dilaton_field
)

from harmonic_sweep import harmonic_sweep

from config import PerturbationConfig
from ft_config import FtOptimalConfig

os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.9'

################################################################################
# Setting up variables
################################################################################

matplotlib.use('TkAgg')

# Full OPTIMIZER_CONFIG
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

# Quick OPTIMIZER_CONFIG
# OPTIMIZER_CONFIG = {
#     "BFGS": True,
#     "Adam (JAX)": False,
#     "Adam (Optax)": False,
#     "Yogi": False,
#     "LBFGS": True,
#     "AdaBelief": True,
#     "Newton's Method": False,
#     "Hessian-based Optimization": False 
# }

def main():
    PertConfig = PerturbationConfig(
        # Core parameters
        T=100.0,                   # Total simulation time
        Z=100.0,                   # Total simulation space
        N=4000,                   # Number of spacetime samples
        G=1,                       # Gravitational constant in 2D
        a=0.1,                     # Stability parameter for back-reaction

        # Fourier perturbation settings
        perturbation_strength=4,   # Perturbation magnitude
        M_user=8,                  # User defined Fourier harmonics
        M_opt=20,                  # Optimizer defined Fourier harmonics

        # Gaussian pulse settings
        pulse_time=0,              # Pulse center
        pulse_amp=0,               # Pulse amplitude
        pulse_width=0              # Pulse width
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

    FtConfig = FtOptimalConfig(results, PertConfig)

################################################################################
# Plotting and Results
################################################################################

    print_optimization_results(results, verbose=False)
    plot_f_vs_ft(results, PertConfig)
    plot_boundary(results, PertConfig)
    plot_dilaton_field(FtConfig , PertConfig)

################################################################################
# Main Function
################################################################################

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "sweep":
        harmonic_sweep()
    else:
        main()
