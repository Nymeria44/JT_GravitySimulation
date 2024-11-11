# harmonic_sweep.py

import pandas as pd
import jax
import os

from config import PerturbationConfig
from schwarzian import run_optimizations, action_to_minimize

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

def harmonic_sweep():
    """
    Sweep over harmonic modes and optimize action.

    Performs optimization across different combinations of user-controlled (M_user)
    and optimizer-controlled (M_opt) harmonics. Records action values and 
    computation times for each enabled optimizer.

    Ranges
    ------
    M_user : [5, 10, 12, 15]
        User-controlled harmonic counts
    M_opt : [10, 20, 30, 40]
        Optimizer-controlled harmonic counts

    Notes
    -----
    Results are saved to 'src/data/harmonic_sweep_results.csv' with columns:
    - M_user, M_opt
    - {optimizer_name} Action
    - {optimizer_name} Time
    """
    # Ensure data directory exists
    data_dir = os.path.join('src', 'data')
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, 'harmonic_sweep_results.csv')

    M_user_range = [5, 10, 12, 15]
    M_opt_range = [10, 20, 30, 40]

    # Get active optimizers
    active_optimizers = [opt for opt, enabled in OPTIMIZER_CONFIG.items() if enabled]
    print(f"Active Optimizers: {active_optimizers}")

    # Setup DataFrame columns
    columns = ["M_user", "M_opt"]
    for optimizer in active_optimizers:
        columns.extend([f"{optimizer} Action", f"{optimizer} Time"])
    results_df = pd.DataFrame(columns=columns)

    for M_user in M_user_range:
        for M_opt in M_opt_range:
            # Configure system
            PertConfig = PerturbationConfig(
                T=10.0,
                Z=10.0,
                N=1000000,
                G=1,
                a=1,
                perturbation_strength=0.1,
                M_user=M_user,
                M_opt=M_opt,
                pulse_time=0,
                pulse_amp=0,
                pulse_width=0
            )
            PertConfig.validate_pulse_width()

            # Initialize optimization parameters
            key_opt = jax.random.PRNGKey(0)
            p_initial = jax.random.normal(key_opt, shape=(2 * M_opt,))

            def objective_function(p):
                return action_to_minimize(p, PertConfig)

            # Run optimizations
            results = run_optimizations(
                action_to_minimize=objective_function,
                p_initial=p_initial,
                config=OPTIMIZER_CONFIG
            )

            # Record results
            row_data = {
                "M_user": M_user,
                "M_opt": M_opt
            }
            for optimizer in active_optimizers:
                action = results["action_values"].get(optimizer, float('nan'))
                time_taken = results["times_taken"].get(optimizer, float('nan'))
                row_data[f"{optimizer} Action"] = action
                row_data[f"{optimizer} Time"] = time_taken

            # Update DataFrame
            row_df = pd.DataFrame([row_data])
            results_df = pd.concat([results_df, row_df], ignore_index=True)
            print(f"Results for M_user={M_user}, M_opt={M_opt} recorded.")

    results_df.to_csv(output_path, index=False)
    print(f"Harmonic sweep completed and results saved to {output_path}")
