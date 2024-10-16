# __init__.py
import numpy as np
from sympy import symbols, sin
from schwarzian import schwarzian_action

# Define symbolic variables
t = symbols('t')
epsilon, omega = symbols('epsilon omega')

# Parameters for small perturbation
epsilon_value = 0.01  # Small perturbation parameter
omega_value = 2 * np.pi  # Frequency of the perturbation

# Define the perturbation function and reparametrization
delta_f_sym = sin(omega * t)            # Perturbation function δf(t)
f_sym = t + epsilon * delta_f_sym       # Reparametrized boundary function f(t)

# Integration limits
t0 = 0
t1 = 1

# Compute the Schwarzian action for the perturbed boundary
action = schwarzian_action(
    f_sym, t, t0, t1, C=1, numerical=True, subs={epsilon: epsilon_value, omega: omega_value}
)

# Output the Schwarzian action result
print(f"Schwarzian Action for the perturbed boundary: {action}")
