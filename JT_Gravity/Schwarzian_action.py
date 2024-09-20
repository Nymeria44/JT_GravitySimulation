import numpy as np
import matplotlib.pyplot as plt

# Define constants from the notes
epsilon = 1e-3  # UV cutoff (Fefferman-Graham gauge)
a = 1.0  # Dimensional parameter controlling dilaton at the boundary
G = 1.0  # Gravitational constant

# Define time range for the boundary curve F(t)
t_min, t_max = 0, 10000
t = np.linspace(t_min, t_max, 1000)

# Define the boundary curve F(t) and its derivatives
# Boundary curve could be any desired reparametrization of time.
# Here we use a perturbed linear function as an example.
F = t + epsilon * np.sin(t)  # Simple perturbation of linear time

# Calculate the first, second, and third derivatives of F(t)
F_dot = np.gradient(F, t)
F_ddot = np.gradient(F_dot, t)
F_dddot = np.gradient(F_ddot, t)

# Boundary condition: Z(t) = epsilon * F'(t)
Z = epsilon * F_dot

# Boundary dilaton condition: Φ_bdy = a / (2 * epsilon)
Phi_bdy = a / (2 * epsilon)

# Schwarzian derivative {F, t}
Schwarzian = (F_dddot / F_dot) - (3 / 2) * (F_ddot / F_dot)**2

# Compute the Hawking boundary parameters (as per the notes)
# √(-γ) = 1/epsilon, K = 1 + epsilon^2 * {F(t), t}
gamma = 1 / epsilon
K = 1 + epsilon**2 * Schwarzian

# Compute the action S = -C * ∫ dt {F, t}
C = a / (16 * np.pi * G)

# Numerical integration over time for the Schwarzian action
Schwarzian_action = -C * np.trapz(Schwarzian, t)

# Plot the Schwarzian derivative
plt.figure(figsize=(8, 6))
plt.plot(t, Schwarzian, label='Schwarzian {F(t), t}')
plt.xlabel('t')
plt.ylabel('Schwarzian {F(t), t}')
plt.title('Schwarzian Derivative of the Boundary Curve')
plt.legend()
plt.show()

# Output the computed Schwarzian action
print("Schwarzian Action:", Schwarzian_action)
