import jax
import jax.numpy as jnp
from jax import grad
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt
import os

# Custom Imports from `optimisations.py`
from optimisations import (
    run_bfgs_optimization,
    run_adam_optimization,
    run_newtons_method,
    run_hessian_optimization,
    print_final_comparison
)

# Constants and setup
os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'

C = 1.0    # Gravitational coupling constant
T = 100.0  # Period for integration
N = 100000  # Number of time steps
t = jnp.linspace(0.001, T, N)  # Time grid

# Number of basis functions
M = 50
n = jnp.arange(1, M + 1)  # Frequencies from 1 to M

# Initial parameters (small random perturbation)
key = jax.random.PRNGKey(0)
p_initial = jax.random.normal(key, shape=(2 * M,)) * 0.01  # Updated to include both sine and cosine coefficients

# Define f(t, p) with both sine and cosine terms
def f(t, p):
    M = p.shape[0] // 2
    sin_coeffs = p[:M]
    cos_coeffs = p[M:]
    sin_terms = jnp.sin(2 * jnp.pi * n[:, None] * t[None, :] / T)  # Shape (M, N)
    cos_terms = jnp.cos(2 * jnp.pi * n[:, None] * t[None, :] / T)  # Shape (M, N)
    delta_f = jnp.dot(sin_coeffs, sin_terms) + jnp.dot(cos_coeffs, cos_terms)  # Shape (N,)
    return t + delta_f  # Shape (N,)

# First derivative f'(t)
def f_prime(t, p):
    M = p.shape[0] // 2
    sin_coeffs = p[:M]
    cos_coeffs = p[M:]
    sin_deriv = jnp.cos(2 * jnp.pi * n[:, None] * t[None, :] / T)
    cos_deriv = -jnp.sin(2 * jnp.pi * n[:, None] * t[None, :] / T)
    delta_f_prime = jnp.dot(sin_coeffs * (2 * jnp.pi * n / T), sin_deriv) + jnp.dot(cos_coeffs * (2 * jnp.pi * n / T), cos_deriv)
    return 1 + delta_f_prime

# Second derivative f''(t)
def f_double_prime(t, p):
    M = p.shape[0] // 2
    sin_coeffs = p[:M]
    cos_coeffs = p[M:]
    sin_double_deriv = -jnp.sin(2 * jnp.pi * n[:, None] * t[None, :] / T)
    cos_double_deriv = -jnp.cos(2 * jnp.pi * n[:, None] * t[None, :] / T)
    delta_f_double_prime = jnp.dot(sin_coeffs * ((2 * jnp.pi * n / T) ** 2), sin_double_deriv) + jnp.dot(cos_coeffs * ((2 * jnp.pi * n / T) ** 2), cos_double_deriv)
    return delta_f_double_prime

# Third derivative f'''(t)
def f_triple_prime(t, p):
    M = p.shape[0] // 2
    sin_coeffs = p[:M]
    cos_coeffs = p[M:]
    sin_triple_deriv = -jnp.cos(2 * jnp.pi * n[:, None] * t[None, :] / T)
    cos_triple_deriv = jnp.sin(2 * jnp.pi * n[:, None] * t[None, :] / T)
    delta_f_triple_prime = jnp.dot(sin_coeffs * ((2 * jnp.pi * n / T) ** 3), sin_triple_deriv) + jnp.dot(cos_coeffs * ((2 * jnp.pi * n / T) ** 3), cos_triple_deriv)
    return delta_f_triple_prime

# Define the Schwarzian derivative
def schwarzian_derivative(t, p):
    fp = f_prime(t, p)
    fpp = f_double_prime(t, p)
    fppp = f_triple_prime(t, p)
    S = fppp / fp - 1.5 * (fpp / fp) ** 2
    return S

# Trapezoidal integration
def jax_trapz(y, x):
    dx = jnp.diff(x)
    return jnp.sum((y[:-1] + y[1:]) * dx / 2.0)

# Define the Schwarzian action
def schwarzian_action(p):
    S = schwarzian_derivative(t, p)
    action = -C * jax_trapz(S, t)
    return action

# Objective function to minimize
def action_to_minimize(p):
    return schwarzian_action(p)

# Compute gradient and Hessian of the action
grad_action = grad(action_to_minimize)
hessian_action = jax.hessian(action_to_minimize)

# Run and time each optimisation method
bfgs_result, bfgs_time = run_bfgs_optimization(action_to_minimize, p_initial)
adam_result, adam_time = run_adam_optimization(action_to_minimize, p_initial)
newton_result, newton_time = run_newtons_method(action_to_minimize, grad_action, hessian_action, p_initial)
hessian_result, hessian_time = run_hessian_optimization(action_to_minimize, grad_action, hessian_action, p_initial)

# Final Comparison
print_final_comparison(bfgs_result, bfgs_time, adam_result, adam_time, newton_result, newton_time, hessian_result, hessian_time)

# Plotting the results for one of the methods (e.g., Adam)
f_optimal = f(t, p_initial)  # Note: Adjust `p_initial` to the best performing result if needed
f_t_minus_t = f_optimal - t  # Deviation from linearity

# Plot the optimized f(t)
plt.plot(t, f_optimal, label="Optimized f(t) using Adam")
plt.xlabel("t")
plt.ylabel("f(t)")
plt.title("Optimized Reparametrisation of f(t) using Adam")
plt.legend()
plt.show()

# Plot the deviation from linearity
plt.plot(t, f_t_minus_t, label="Deviation from Linearity (f(t) - t)")
plt.xlabel("t")
plt.ylabel("f(t) - t")
plt.title("Deviation of Optimized f(t) from Linearity using Adam")
plt.legend()
plt.show()
