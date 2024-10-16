import numpy as np
import matplotlib.pyplot as plt

# Define the range for F and Z
F_min, F_max = -10, 10
Z_min, Z_max = 0.1, 10 

# Create an array of values for F and Z
F = np.linspace(F_min, F_max, 400)
Z = np.linspace(Z_min, Z_max, 400)
F, Z = np.meshgrid(F, Z)

# Lightcone coordinates
U = F + Z
V = F - Z

# Compute differentials dU and dV
dU = np.gradient(U, axis=1)  # Differentiate along the F axis
dV = np.gradient(V, axis=0)  # Differentiate along the Z axis

# Handle small numerical errors (avoid division by zero)
epsilon = 1e-10  # Small number to avoid division by exactly zero

# Compute ds^2 using the metric in the Poincaré patch
with np.errstate(divide='ignore', invalid='ignore'):
    ds2_poincare = -4 * dU * dV / np.where(np.abs(U - V) < epsilon, epsilon, (U - V)**2)
    ds2_poincare[np.isinf(ds2_poincare) | np.isnan(ds2_poincare)] = np.nan  # Handle infinities and NaNs

# Plot Poincaré patch metric
fig = plt.figure(figsize=(8, 6))
ax1 = fig.add_subplot(111, projection='3d')
surf1 = ax1.plot_surface(F, Z, ds2_poincare, cmap='viridis')
ax1.set_xlabel('F')
ax1.set_ylabel('Z')
ax1.set_zlabel('ds^2')
ax1.set_title('Poincaré Patch in Lightcone Coordinates')

# Adjust the layout and show the plot
plt.tight_layout()
plt.show()
