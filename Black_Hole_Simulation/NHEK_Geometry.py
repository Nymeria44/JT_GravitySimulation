import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Define the coordinates
r, theta, phi = sp.symbols('r theta phi')

# Define the constants
G = sp.Symbol('G')  # Gravitational constant
J = sp.Symbol('J')  # Angular momentum

# Define differentials
dr = sp.diff(r)
dtheta = sp.diff(theta)
dphi = sp.diff(phi)

# Define the metric components
Omega_sq = (1 + sp.cos(theta)**2) / 2
Lambda = 2 * sp.sin(theta) / (1 + sp.cos(theta)**2)

# Define ds^2
ds_squared = 2*G*J*Omega_sq*(dr**2 + r**2 * dtheta**2 + r**2 * sp.sin(theta)**2 * dphi**2)

# Convert the symbolic expression for ds^2 to a NumPy function
ds_squared_func = sp.lambdify((r, theta, phi), ds_squared.subs({G: 1, J: 1}), modules=['numpy'])

# Define the range for the variables
r_vals = np.linspace(0.01, 10, 100)
theta_vals = np.linspace(0, np.pi, 100)
phi_vals = np.linspace(0, 2*np.pi, 100)

# Create meshgrid for the variables
R, THETA, PHI = np.meshgrid(r_vals, theta_vals, phi_vals, indexing='ij')

# Calculate ds^2 values on the meshgrid
ds_squared_vals = ds_squared_func(R, THETA, PHI)

# Convert spherical to Cartesian coordinates
X = (R * np.sin(THETA) * np.cos(PHI)).flatten()
Y = (R * np.sin(THETA) * np.sin(PHI)).flatten()
Z = (R * np.cos(THETA)).flatten()
ds_squared_vals_flat = ds_squared_vals.flatten()

# Plot the geometry
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
ax.plot_trisurf(X, Y, Z, cmap=cm.Blues, linewidth=0.2)

# Set labels and title
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('NHEK Geometry')

plt.show()
