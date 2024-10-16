# __init__.py
import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, sin, lambdify
from schwarzian import schwarzian_action

# Define symbolic variables
t = symbols('t')
epsilon, omega = symbols('epsilon omega')

# Parameters for small perturbation
epsilon_value = 0.01  # Small perturbation parameter
omega_value = 2 * np.pi  # Frequency of the perturbation

# Define the unperturbed boundary reparametrization
f_unperturbed = t  # Unperturbed boundary function f(t) = t

# Define the perturbed boundary reparametrization
# delta_f_sym = sin(omega * t)             
f_perturbed = t + epsilon * 50000

# Integration limits
t0 = 0
t1 = 1

# Compute the Schwarzian action for the unperturbed boundary
action_unperturbed = schwarzian_action(
    f_unperturbed, t, t0, t1, C=1, numerical=True
)

# Compute the Schwarzian action for the perturbed boundary
action_perturbed = schwarzian_action(
    f_perturbed, t, t0, t1, C=1, numerical=True, subs={epsilon: epsilon_value, omega: omega_value}
)

# Output the results for comparison
print(f"Schwarzian Action for the unperturbed boundary: {action_unperturbed}")
print(f"Schwarzian Action for the perturbed boundary: {action_perturbed}")
print(f"Difference due to perturbation: {action_perturbed - action_unperturbed}")

# Substitute numerical values into the perturbed and unperturbed functions
f_perturbed_numeric = f_perturbed.subs({epsilon: epsilon_value, omega: omega_value})
f_perturbed_func = lambdify(t, f_perturbed_numeric, modules=['numpy'])
f_unperturbed_func = lambdify(t, f_unperturbed, modules=['numpy'])

# Generate time values for plotting
t_values = np.linspace(t0, t1, 1000)

# Calculate the boundary reparametrizations
f_perturbed_values = f_perturbed_func(t_values)
f_unperturbed_values = f_unperturbed_func(t_values)

# Plot the perturbed and unperturbed boundary reparametrizations
plt.plot(t_values, f_unperturbed_values, label='Unperturbed Boundary f(t) = t')
plt.plot(t_values, f_perturbed_values, label=f'Perturbed Boundary f(t) = t + ε * sin(ωt), ε = {epsilon_value}')
plt.xlabel('t')
plt.ylabel('f(t)')
plt.title('Boundary Reparametrization: Perturbed vs. Unperturbed')
plt.legend()
plt.grid(True)
plt.show()
