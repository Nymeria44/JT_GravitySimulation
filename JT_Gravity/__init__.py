# __init__.py file for JT_Gravity package
import jax
import matplotlib
import os

from schwarzian import (
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
    "BFGS": False,
    "Adam (JAX)": False,
    "Adam (Optax)": False,
    "Yogi": True,
    "LBFGS": False,
    "AdaBelief": True,
    "Newton's Method": False,
    "Hessian-based Optimization": False
}

def main():
    # Initialize perturbation configuration, including user and optimizer parameters
    PertConfig = PerturbationConfig(
        # Core parameters
        T=10.0,                   # Total sim time
        N=10000,                  # Number of time samples
        C=1.0,                    # Scaling factor of action

        # Fourier perturbation settings
        perturbation_strength=30, # Magnitude of user Fourier Pertubation
        M_user=0,                 # Number of Fourier modes for user-defined perturbation
        M_opt=100,                # Number of Fourier modes controlled by optimiser

        # Gaussian pulse settings
        pulse_time=5,             # Center of Gaussian pulse
        pulse_amp=1000,              # Amplitude of Gaussian pulse
        pulse_width=0.1             # Width of Gaussian pulse
    )
    PertConfig.validate_pulse_width() # Checking pulse width isn't too small to be detected

    # Initial guess for optimizer-controlled parameters
    key_opt = jax.random.PRNGKey(0)
    p_initial = jax.random.normal(key_opt, shape=(2 * PertConfig .M_opt,)) * 0.01

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
    main()
