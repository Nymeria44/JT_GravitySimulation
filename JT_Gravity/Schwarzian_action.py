import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Constants
C = 1.0  # Coupling constant

# Time parameters
t_start = 0.0
t_end = 10.0
N = 1000
t = np.linspace(t_start, t_end, N)

# Initial perturbation
delta = 0.01
phi_initial = delta * np.sin(2 * np.pi * t / (t_end - t_start))

# Initial conditions for phi and its derivatives
phi0 = phi_initial[0]
phi_prime0 = np.gradient(phi_initial, t)[0]
phi_double_prime0 = np.gradient(np.gradient(phi_initial, t), t)[0]
phi_triple_prime0 = np.gradient(np.gradient(np.gradient(phi_initial, t), t), t)[0]
y0 = [phi0, phi_prime0, phi_double_prime0, phi_triple_prime0]

# Equations of motion
def equations_of_motion(t, y):
    phi, phi_prime, phi_double_prime, phi_triple_prime = y
    phi_quadruple_prime = phi_triple_prime + phi_double_prime  # From phi'''' - phi''' = 0
    return [phi_prime, phi_double_prime, phi_triple_prime, phi_quadruple_prime]

# Solve the ODE system
sol = solve_ivp(
    equations_of_motion,
    [t_start, t_end],
    y0,
    t_eval=t,
    method='RK45',
    atol=1e-8,
    rtol=1e-8
)

# Extract the solution
phi_evolved = sol.y[0]
f_evolved = t + phi_evolved

# Compute Schwarzian derivative
def schwarzian_derivative(f, t):
    f_prime = np.gradient(f, t)
    f_double_prime = np.gradient(f_prime, t)
    f_triple_prime = np.gradient(f_double_prime, t)
    epsilon = 1e-8
    f_prime_safe = np.where(np.abs(f_prime) < epsilon, epsilon, f_prime)
    s = (f_triple_prime / f_prime_safe) - (1.5) * (f_double_prime / f_prime_safe)**2
    return s

s_initial = schwarzian_derivative(t + phi_initial, t)
s_evolved = schwarzian_derivative(f_evolved, t)

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t, phi_initial, label=r'Initial $\phi(t)$')
plt.plot(t, phi_evolved, label=r'Evolved $\phi(t)$')
plt.xlabel('Time $t$')
plt.ylabel(r'$\phi(t)$')
plt.title(r'Evolution of the Perturbation $\phi(t)$')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t, s_initial, label=r'Initial Schwarzian $\{f(t), t\}$')
plt.plot(t, s_evolved, label=r'Evolved Schwarzian $\{f(t), t\}$')
plt.xlabel('Time $t$')
plt.ylabel('Schwarzian Derivative')
plt.title('Schwarzian Derivative Before and After Evolution')
plt.legend()

plt.tight_layout()
plt.show()
