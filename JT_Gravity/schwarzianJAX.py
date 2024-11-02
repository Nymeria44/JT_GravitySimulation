import jax
import jax.numpy as jnp
from jax import grad
from jax.scipy.optimize import minimize
from jax.example_libraries import optimizers
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
M = 20
n = jnp.arange(1, M + 1)  # Frequencies from 1 to M

# Initial parameters (small random perturbation)
key = jax.random.PRNGKey(0)
p_initial = jax.random.normal(key, shape=(2*M,)) * 0.01  # Updated to include both sine and cosine coefficients

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

# 1. BFGS Optimization
result_bfgs = minimize(action_to_minimize, p_initial, method='BFGS')
p_optimal_bfgs = result_bfgs.x  # Optimal parameters from BFGS
action_value_bfgs = action_to_minimize(p_optimal_bfgs)
print("BFGS Optimization Complete.")
print(f"BFGS Optimized Action: {action_value_bfgs}")

# 2. Adam Optimization
opt_init, opt_update, get_params = optimizers.adam(step_size=0.01)
opt_state = opt_init(p_initial)

@jax.jit
def adam_step(i, opt_state):
    p = get_params(opt_state)
    value, grads = jax.value_and_grad(action_to_minimize)(p)
    opt_state = opt_update(i, grads, opt_state)
    return opt_state, value

num_steps = 1000
for i in range(num_steps):
    opt_state, action_value_adam = adam_step(i, opt_state)
    if i % 100 == 0:
        print(f"Adam Step {i}, Action Value: {action_value_adam}")

p_optimal_adam = get_params(opt_state)
print("Adam Optimization Complete.")
print(f"Adam Optimized Action: {action_value_adam}")

# 3. Newton's Method Optimization
p_newton = p_initial
num_newton_steps = 10
for i in range(num_newton_steps):
    grad_val = grad_action(p_newton)
    hess_val = hessian_action(p_newton)
    hess_val += jnp.eye(hess_val.shape[0]) * 1e-6  # Regularization
    delta_p = jnp.linalg.solve(hess_val, grad_val)
    p_newton = p_newton - delta_p
    action_value_newton = action_to_minimize(p_newton)
    print(f"Newton Step {i}, Action Value: {action_value_newton}")

print("Newton's Method Optimization Complete.")
print(f"Newton's Method Optimized Action: {action_value_newton}")

# 4. Hessian-based Optimization
p_hessian = p_initial
num_hessian_steps = 20
learning_rate = 0.1
for i in range(num_hessian_steps):
    grad_val = grad_action(p_hessian)
    hess_val = hessian_action(p_hessian)
    hess_val += jnp.eye(hess_val.shape[0]) * 1e-6  # Regularization
    delta_p = jnp.linalg.solve(hess_val, grad_val)
    p_hessian = p_hessian - learning_rate * delta_p
    action_value_hessian = action_to_minimize(p_hessian)
    print(f"Hessian Step {i}, Action Value: {action_value_hessian}")

print("Hessian-based Optimization Complete.")
print(f"Hessian-based Optimized Action: {action_value_hessian}")

# Final Action Values Comparison
print("\nFinal Action Values Comparison:")
print(f"BFGS: {action_value_bfgs}")
print(f"Adam: {action_value_adam}")
print(f"Newton's Method: {action_value_newton}")
print(f"Hessian-based Optimization: {action_value_hessian}")

# Plotting the results for one of the methods (e.g., Adam)
f_optimal = f(t, p_optimal_adam)
f_t_minus_t = f_optimal - t  # Deviation from linearity

# Plot the optimized f(t)
plt.plot(t, f_optimal, label="Optimized f(t) using Adam")
plt.xlabel("t")
plt.ylabel("f(t)")
plt.title("Optimized Reparametrization of f(t) using Adam")
plt.legend()
plt.show()

# Plot the deviation from linearity
plt.plot(t, f_t_minus_t, label="Deviation from Linearity (f(t) - t)")
plt.xlabel("t")
plt.ylabel("f(t) - t")
plt.title("Deviation of Optimized f(t) from Linearity using Adam")
plt.legend()
plt.show()
