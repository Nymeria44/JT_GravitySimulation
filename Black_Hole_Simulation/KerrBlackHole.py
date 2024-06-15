import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, pi, sin, sqrt, diff, roots
from numpy import meshgrid

#----------------------------------------
# PARAMETERS
#----------------------------------------
simRes = 0.05  # Defines the minimum interval between coordinates (from 0 to 1)

# NOTE: the spacetime metric is only valid for small black holes
M = 4000  # Mass of Black Hole (KG)

#----------------------------------------
# FUNCTIONS
#----------------------------------------
# Determines the radius of the event horizon according to the Schwarzschild Radius
def calc_bh_radius(M):
    c = 2.99792458E8  # Speed of light
    G = 6.67408E-11  # Gravitational Constant

    rs = 2 * G * M / c ** 2  # Schwarzschild radius
    return rs

# Generates spacetime coordinates and their interval
def generate_coordinates(simRes, R):
    # Create radial and angular coordinates
    r = np.arange(0, 1, simRes) * R
    t = np.arange(0, 1, simRes) 
    theta = np.arange(0, 1, simRes) * np.pi
    phi = np.arange(0, 1, simRes) * 2 * np.pi

    # Calculate differential steps
    dr = np.diff(r)
    dt = np.diff(t)
    dtheta = np.diff(theta)
    dphi = np.diff(phi)

    # Duplicate the last element of each array to avoid incompatible sizes
    dr = np.append(dr, dr[-1])
    dt = np.append(dt, dt[-1])
    dtheta = np.append(dtheta, dtheta[-1])
    dphi = np.append(dphi, dphi[-1])

    return r, t, theta, phi, dr, dt, dtheta, dphi

# Determines spacetime interval
def space_time_interval(r, R, M, theta, dt, dr, dtheta, dphi):
    G = 6.67408E-11  # Gravitational Constant

    f_r = (1 - (r ** 2 / R ** 2) - (2 * M * G / r))
    
    ds_sqr = -f_r * dt ** 2 + (dr ** 2 / f_r) * r ** 2 * (dtheta ** 2 + (np.sin(theta)) ** 2 * dphi ** 2)
    ds = np.sqrt(ds_sqr)
    return ds

# Converts spherical coordinates to cartesian coordinates
def sph_to_cart(r, theta):
    R_mesh, Theta_mesh = np.meshgrid(r, theta)
    x = R_mesh * np.sin(Theta_mesh)
    y = R_mesh * np.sin(Theta_mesh)  # Corrected calculation
    return x, y

# Finds value of r at horizons
def find_horizons(R, M):
    G = 6.67408E-11  # Gravitational Constant

    # Solving f_r equation for roots
    horizon_coefficients = [(-1 / R ** 2), -2 * M * G, 1]
    r_h = np.roots(horizon_coefficients)

    # Filtering for real and positive roots
    r_h = r_h[np.isreal(r_h) & (r_h > 0)]
    return r_h

#----------------------------------------
# CALCULATIONS
#----------------------------------------
rs = calc_bh_radius(M)  # Finding Schwarzschild radius
R = rs * 10E10  # Radius of the de Sitter space relative to black hole

# Generating radial coordinates
r, t, theta, phi, dr, dt, dtheta, dphi = generate_coordinates(simRes, R)

# Calculating spacetime interval
ds = space_time_interval(r, R, M, theta, dt, dr, dtheta, dphi)

#----------------------------------------
# PLOT SPACETIME
#----------------------------------------
# Converting radial coordinates into Cartesian
x, y = sph_to_cart(r, theta)
ds_matrix = np.meshgrid(ds)

# Assuming ds_matrix is a list, convert it to a 2D NumPy array
ds_matrix = np.array(ds_matrix)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

Spacetime = ax.plot_surface(x, y, ds_matrix, cmap='jet')

# Dotted lines on mesh for better visibility
Spacetime.set_linestyle(':')

#----------------------------------------
# PLOT HORIZONS
#----------------------------------------
# finding horizon of de Sitter space
r_h = find_horizons(R, M)
ds_h = space_time_interval(r_h, R, M, theta[0], dt[0], dr[0], dtheta[0], dphi[0])
x_h, y_h = sph_to_cart(r_h, theta[0])

deSitterHorizon = ax.plot(x_h, y_h, ds_h, 'r', linewidth=4, label='de Sitter Horizon')

# legend, title and labels
ax.legend()
ax.set_title('Visualization of Schwarzschild-de Sitter Black Hole')
ax.set_xlabel('x (spatial dimension)')
ax.set_ylabel('y (spatial dimension)')
ax.set_zlabel('Spacetime Interval (ds)')

# Set viewing position
# ax.view_init(elev=20, azim=25)  # Camera tilt

plt.show()
