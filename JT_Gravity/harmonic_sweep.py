# harmonic_sweep.py

import pandas as pd
import jax
from config import PerturbationConfig
from schwarzian import run_optimizations, action_to_minimize

# Configuration dictionary to enable/disable specific optimizers
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
    Performs a harmonic sweep by varying M_user and M_opt, running optimizations
    with the enabled optimizers, and recording the results.
    """
    # Define harmonic sweep ranges
    # M_user_range = [5, 10, 12, 15]
    # M_opt_range = [10, 20, 30, 40]
    M_user_range = [5]
    M_opt_range = [10]

    # Dynamically determine active optimizers
    active_optimizers = [opt for opt, enabled in OPTIMIZER_CONFIG.items() if enabled]
    print(f"Active Optimizers: {active_optimizers}")

    # Initialize DataFrame columns dynamically
    columns = ["M_user", "M_opt"]
    for optimizer in active_optimizers:
        columns.extend([f"{optimizer} Action", f"{optimizer} Time"])

    # Initialize DataFrame to store results
    results_df = pd.DataFrame(columns=columns)

    for M_user in M_user_range:
        for M_opt in M_opt_range:
            # Initialize PerturbationConfig with correct harmonic assignment
            PertConfig = PerturbationConfig(
                T=10.0,
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

            # Initialise random key and initial perturbation parameters
            key_opt = jax.random.PRNGKey(0)
            p_initial = jax.random.normal(key_opt, shape=(2 * M_opt,))

            def objective_function(p):
                return action_to_minimize(p, PertConfig)

            # Run optimizations with the active optimizers
            results = run_optimizations(
                action_to_minimize=objective_function,
                p_initial=p_initial,
                config=OPTIMIZER_CONFIG
            )

            # Prepare row data dynamically based on active optimizers
            row_data = {
                "M_user": M_user,
                "M_opt": M_opt
            }

            for optimizer in active_optimizers:
                action = results["action_values"].get(optimizer, float('nan'))
                time_taken = results["times_taken"].get(optimizer, float('nan'))
                row_data[f"{optimizer} Action"] = action
                row_data[f"{optimizer} Time"] = time_taken

            # Convert row_data to DataFrame and concatenate
            row_df = pd.DataFrame([row_data])
            results_df = pd.concat([results_df, row_df], ignore_index=True)
            print(f"Results for M_user={M_user}, M_opt={M_opt} recorded.")

    # Save the results to CSV
    results_df.to_csv("harmonic_sweep_results.csv", index=False)
    print("Harmonic sweep completed and results saved to harmonic_sweep_results.csv.")
