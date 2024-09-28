import numpy as np
import matplotlib.pyplot as plt

# Constants
C = 1.0  # Coupling constant in the Schwarzian action

# Time parameters
t_start = 0.0
t_end = 10.0
N = 1000  # Number of time points
t = np.linspace(t_start, t_end, N)
dt = t[1] - t[0]

# Initialize f(t) with a small perturbation
epsilon = 0.01
f = t + epsilon * np.sin(2 * np.pi * t / (t_end - t_start))

# Function to compute derivatives
def compute_derivatives(f, dt):
    # First derivative
    f_prime = np.zeros_like(f)
    f_prime[1:-1] = (f[2:] - f[:-2]) / (2 * dt)
    f_prime[0] = (f[1] - f[0]) / dt  # Forward difference
    f_prime[-1] = (f[-1] - f[-2]) / dt  # Backward difference

    # Second derivative
    f_double_prime = np.zeros_like(f)
    f_double_prime[1:-1] = (f[2:] - 2 * f[1:-1] + f[:-2]) / dt**2
    f_double_prime[0] = (f[2] - 2 * f[1] + f[0]) / dt**2
    f_double_prime[-1] = (f[-1] - 2 * f[-2] + f[-3]) / dt**2

    # Third derivative
    f_triple_prime = np.zeros_like(f)
    f_triple_prime[2:-2] = (f[4:] - 2 * f[3:-1] + 2 * f[1:-3] - f[0:-4]) / (2 * dt**3)
    f_triple_prime[0:2] = f_triple_prime[2]
    f_triple_prime[-2:] = f_triple_prime[-3]

    return f_prime, f_double_prime, f_triple_prime

# Function to compute Schwarzian derivative
def schwarzian_derivative(f, dt):
    f_prime, f_double_prime, f_triple_prime = compute_derivatives(f, dt)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        s = (f_triple_prime / f_prime) - (1.5) * (f_double_prime / f_prime)**2
        s = np.nan_to_num(s)  # Replace NaNs and infs with zero
    return s

# Time evolution using Euler method
def evolve_f(f, dt, steps):
    f_evolved = np.copy(f)
    for step in range(steps):
        s = schwarzian_derivative(f_evolved, dt)
        # Equation of motion: C * d^2 f / dt^2 = variation of Schwarzian action
        # For simplicity, assume d^2 f / dt^2 = -delta S / delta f
        # Here, we approximate delta S / delta f ~ Schwarzian derivative
        acceleration = -C * s
        # Update f using simple integration (Euler method)
        # Need to keep track of velocities (first derivatives)
        if step == 0:
            # Initialize velocity
            velocity = np.zeros_like(f)
        else:
            # Update velocity and position
            velocity += acceleration * dt
            f_evolved += velocity * dt
    return f_evolved

# Number of evolution steps
steps = 100

# Evolve f(t)
f_evolved = evolve_f(f, dt, steps)

# Compute Schwarzian derivative
s_initial = schwarzian_derivative(f, dt)
s_evolved = schwarzian_derivative(f_evolved, dt)

# Plot results
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t, f, label='Initial f(t)')
plt.plot(t, f_evolved, label='Evolved f(t)')
plt.xlabel('t')
plt.ylabel('f(t)')
plt.title('Evolution of f(t)')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, s_initial, label='Initial Schwarzian {f(t), t}')
plt.plot(t, s_evolved, label='Evolved Schwarzian {f(t), t}')
plt.xlabel('t')
plt.ylabel('Schwarzian Derivative')
plt.title('Schwarzian Derivative Before and After Evolution')
plt.legend()

plt.tight_layout()
plt.show()

