import numpy as np
import matplotlib.pyplot as plt

# Number of points
data_points = 1000

# Time and radial coordinates
T_vals = np.linspace(0, 10, data_points) 
Z_vals = np.linspace(0.01, 10, data_points)
T, Z = np.meshgrid(T_vals, Z_vals)

# Lightcone coordinates U and V
U = T + Z
V = T - Z

# Removing values where U <= V (outside the lightcone)
mask = U > V
U = np.where(mask, U, np.nan)
V = np.where(mask, V, np.nan)

# Calculating metric for T, Z coordinates (ds^2 = (-dT^2 + dZ^2) / Z^2)
g_TT = -1 / Z**2
g_ZZ = 1 / Z**2

# Calculating metric for U, V coordinates (ds^2 = 4 dU dV / (U - V)^2)
g_UV = 4 / (U - V)**2

# Set up the plotting environment
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# Plot the (T, Z) metric components
c1 = axs[0].contourf(T, Z, g_TT, levels=100, cmap='coolwarm')
c2 = axs[0].contourf(T, Z, g_ZZ, levels=100, cmap='coolwarm', alpha=0.5)

axs[0].set_title('Metric in $(T, Z)$ Coordinates')
axs[0].set_xlabel('$T$')
axs[0].set_ylabel('$Z$')
fig.colorbar(c1, ax=axs[0], label='$g_{TT}$')
fig.colorbar(c2, ax=axs[0], label='$g_{ZZ}$')

# Plot the (U, V) metric component
c3 = axs[1].contourf(U, V, g_UV, levels=100, cmap='viridis')

axs[1].set_title('Metric in Lightcone $(U, V)$ Coordinates')
axs[1].set_xlabel('$U$')
axs[1].set_ylabel('$V$')
fig.colorbar(c3, ax=axs[1], label='$g_{UV}$')

# Adjust layout and display the plots
plt.tight_layout()
plt.show()
