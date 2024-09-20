import numpy as np
import matplotlib.pyplot as plt

# Define the range for F and Z
F_min, F_max = -10, 10
Z_min, Z_max = 0.1, 10 

# Create an array of values for F and Z
F = np.linspace(F_min, F_max, 400)
Z = np.linspace(Z_min, Z_max, 400)

# Create a grid of F and Z values
F, Z = np.meshgrid(F, Z)

# Calculate differentials
dF = np.gradient(F, axis=1)  # Differentiate along the F axis
dZ = np.gradient(Z, axis=0)  # Differentiate along the Z axis

# Compute spacetime curvature according to ds^2 = (-dF^2 + dZ^2) / Z^2
ds2 = (-dF**2 + dZ**2) / Z**2

# Plot the surface
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(F, Z, ds2, cmap='viridis')

# Add labels and title
ax.set_xlabel('F')
ax.set_ylabel('Z')
ax.set_zlabel('ds^2')
ax.set_title('JT Gravity Metric Visualization')

# Add a color bar to represent the scale
fig.colorbar(surf, shrink=0.5, aspect=5)

# Show the plot
plt.show()

