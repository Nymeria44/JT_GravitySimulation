import jax
import jax.numpy as jnp
from jax import grad
from jax.scipy.optimize import minimize
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt

# Constants and setup
import os
os.environ['XLA_FLAGS'] = '--xla_gpu_triton_gemm_any=True'

jax.config.update("jax_enable_x64", True)

C = 1.0    # Gravitational coupling constant
T = 100.0  # Period for integration
N = 10000   # Number of time steps
t = jnp.linspace(0.001, T, N)  # Time grid

# Number of basis functions
M = 10
n = jnp.arange(1, M + 1)  # Frequencies from 1 to M

# Initial parameters (small random perturbation)
key = jax.random.PRNGKey(0)
p_initial = jax.random.normal(key, shape=(M,)) * 0.01

# Define f(t, p)
def f(t, p):
    sin_terms = jnp.sin(2 * jnp.pi * n[:, None] * t[None, :] / T)  # Shape (M, N)
    delta_f = jnp.dot(p, sin_terms)  # Shape (N,)
    return t + delta_f  # Shape (N,)

# First derivative f'(t)
def f_prime(t, p):
    cos_terms = jnp.cos(2 * jnp.pi * n[:, None] * t[None, :] / T)
    delta_f_prime = jnp.dot(p * (2 * jnp.pi * n / T), cos_terms)
    return 1 + delta_f_prime

# Second derivative f''(t)
def f_double_prime(t, p):
    sin_terms = jnp.sin(2 * jnp.pi * n[:, None] * t[None, :] / T)
    delta_f_double_prime = -jnp.dot(p * ((2 * jnp.pi * n / T) ** 2), sin_terms)
    return delta_f_double_prime

# Third derivative f'''(t)
def f_triple_prime(t, p):
    cos_terms = jnp.cos(2 * jnp.pi * n[:, None] * t[None, :] / T)
    delta_f_triple_prime = -jnp.dot(p * ((2 * jnp.pi * n / T) ** 3), cos_terms)
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

# Compute gradient of the action
grad_action = grad(action_to_minimize)

# Perform optimization
result = minimize(action_to_minimize, p_initial, method='BFGS')
p_optimal = result.x  # Optimal parameters
f_optimal = f(t, p_optimal)
f_t_minus_t = f_optimal - t  # Deviation from linearity

print("Optimized parameters p:", p_optimal)
print("Optimized action:", action_to_minimize(p_optimal))

# Plot the optimized f(t)
plt.plot(t, f_optimal, label="Optimized f(t)")
plt.xlabel("t")
plt.ylabel("f(t)")
plt.title("Optimized Reparametrization of f(t)")
plt.legend()
plt.show()

# Plot the deviation from linearity
plt.plot(t, f_t_minus_t, label="Deviation from Linearity (f(t) - t)")
plt.xlabel("t")
plt.ylabel("f(t) - t")
plt.title("Deviation of Optimized f(t) from Linearity")
plt.legend()
plt.show()
