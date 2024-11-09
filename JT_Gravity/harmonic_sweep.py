# harmonic_sweep.py

import pandas as pd
import jax
from config import PerturbationConfig
from schwarzian import run_optimizations, action_to_minimize


# Configuration dictionary to enable/disable specific optimizers
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

def harmonic_sweep():
    # Define harmonic sweep ranges
    M_user_range = [5, 10, 15]
    M_opt_range = [20, 30, 40]

    # Initialize DataFrame to store results
    results_df = pd.DataFrame(columns=[
        "M_user", "M_opt", "Adam (JAX) Action", "Adam (JAX) Time",
        "Adam (Optax) Action", "Adam (Optax) Time", "Yogi Action", "Yogi Time",
        "AdaBelief Action", "AdaBelief Time"
    ])

    for M_user in M_user_range:
        for M_opt in M_opt_range:
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

            key_opt = jax.random.PRNGKey(0)
            p_initial = jax.random.normal(key_opt, shape=(2 * M_opt,))

            def objective_function(p):
                return action_to_minimize(p, PertConfig)

            results = run_optimizations(
                action_to_minimize=objective_function,
                p_initial=p_initial,
                config=OPTIMIZER_CONFIG
            )

            row_data = {
                "M_user": M_user,
                "M_opt": M_opt,
                "Adam (JAX) Action": results["action_values"].get("Adam (JAX)", float('nan')),
                "Adam (JAX) Time": results["times_taken"].get("Adam (JAX)", float('nan')),
                "Adam (Optax) Action": results["action_values"].get("Adam (Optax)", float('nan')),
                "Adam (Optax) Time": results["times_taken"].get("Adam (Optax)", float('nan')),
                "Yogi Action": results["action_values"].get("Yogi", float('nan')),
                "Yogi Time": results["times_taken"].get("Yogi", float('nan')),
                "AdaBelief Action": results["action_values"].get("AdaBelief", float('nan')),
                "AdaBelief Time": results["times_taken"].get("AdaBelief", float('nan')),
            }

            # Convert row_data to DataFrame and concatenate
            row_df = pd.DataFrame([row_data])
            results_df = pd.concat([results_df, row_df], ignore_index=True)
            print(f"Results for M_user={M_user}, M_opt={M_opt} recorded.")

    results_df.to_csv("harmonic_sweep_results.csv", index=False)
    print("Harmonic sweep completed and results saved to harmonic_sweep_results.csv.")
