import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

########################################
# Bulk
########################################
# Creating bulk coordinates
F_min, F_max = -100, 100
Z_min, Z_max = 1, 600

F_bulk = np.linspace(F_min, F_max, 200)
Z_bulk = np.linspace(Z_min, Z_max, 1000)

# Create a grid of F and Z values for the bulk spacetime
F_bulk, Z_bulk = np.meshgrid(F_bulk, Z_bulk)

# Calculate differentials for the bulk
dF_bulk = np.gradient(F_bulk, axis=1)  # Differentiate along the F axis
dZ_bulk = np.gradient(Z_bulk, axis=0)  # Differentiate along the Z axis

# Compute spacetime curvature for AdS_2 according to ds^2 = (-dF^2 + dZ^2) / Z^2
ds2_bulk = (-dF_bulk**2 + dZ_bulk**2) / Z_bulk**2

# Plot the bulk spacetime surface
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(F_bulk, Z_bulk, ds2_bulk, cmap='viridis', alpha=0.7)



########################################
# Boundary
########################################
epsilon = 1e-3  # UV cutoff (Fefferman-Graham gauge)
a = 1.0  # Dimensional parameter controlling dilaton at the boundary
G = 1.0  # Gravitational constant

# Define time range for the boundary curve F(t)
t_boundary = np.linspace(F_min, F_max, 1000)

# Define the boundary curve F(t) and its derivatives
# We use a simple perturbed linear function for the boundary curve F(t)
F_boundary = t_boundary + epsilon * np.sin(t_boundary)  # Perturbation of linear time

# Calculate the first derivative F'(t) for the boundary
F_dot_boundary = np.gradient(F_boundary, t_boundary)

# Boundary condition: Z(t) = epsilon * F'(t)
Z_boundary = epsilon * F_dot_boundary

# Plot the boundary curve on the same plot as the bulk spacetime
ax.plot(F_boundary, Z_boundary, color='r', label='CFT', linewidth=3)

# labels
ax.set_xlabel('F')
ax.set_ylabel('Z')
ax.set_zlabel('ds^2')
ax.set_title('JT Gravity: Bulk AdS_2 Metric and Boundary Curve')
# Add a legend to differentiate the bulk and boundary components
ax.legend()

# Display the combined plot
plt.show()
