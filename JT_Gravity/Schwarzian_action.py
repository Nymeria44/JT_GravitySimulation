import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp

# Constants
C = 1.0  # Coupling constant

# Time parameters
t_start = 0.0
t_end = 100.0
N = 10000
t = np.linspace(t_start, t_end, N)

# Initial perturbation parameters
delta = 0.00001
omega = 2 * np.pi / (t_end - t_start)

# Initialize the symbolic variable
t_sym = sp.symbols('t')

# Convert delta and omega to SymPy types
delta_sym = sp.Float(delta)
omega_sym = sp.Float(omega)

# Define the perturbation function symbolically
phi_sym = delta_sym * sp.sin(omega_sym * t_sym)

# Compute symbolic derivatives
phi_prime_sym = sp.diff(phi_sym, t_sym)
phi_double_prime_sym = sp.diff(phi_prime_sym, t_sym)
phi_triple_prime_sym = sp.diff(phi_double_prime_sym, t_sym)

# Create numerical functions from symbolic expressions
phi_func = sp.lambdify(t_sym, phi_sym, modules='numpy')
phi_prime_func = sp.lambdify(t_sym, phi_prime_sym, modules='numpy')
phi_double_prime_func = sp.lambdify(t_sym, phi_double_prime_sym, modules='numpy')
phi_triple_prime_func = sp.lambdify(t_sym, phi_triple_prime_sym, modules='numpy')

# Evaluate the functions
phi_initial = phi_func(t)
phi_prime = phi_prime_func(t)
phi_double_prime = phi_double_prime_func(t)
phi_triple_prime = phi_triple_prime_func(t)

# Extract initial values
phi0 = phi_initial[0]
phi_prime0 = phi_prime[0]
phi_double_prime0 = phi_double_prime[0]
phi_triple_prime0 = phi_triple_prime[0]
y0 = [phi0, phi_prime0, phi_double_prime0, phi_triple_prime0]

# Equations of motion
def equations_of_motion(t, y):
    phi, phi_prime, phi_double_prime, phi_triple_prime = y
    phi_quadruple_prime = phi_triple_prime  # Corrected equation
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
    f_prime_safe = f_prime + 1e-12 * np.max(np.abs(f_prime))
    s = (f_triple_prime / f_prime_safe) - (1.5) * (f_double_prime / f_prime_safe)**2
    return s

s_initial = schwarzian_derivative(t + phi_initial, t)
s_evolved = schwarzian_derivative(f_evolved, t)

# Downsampling for plotting
downsample_factor = 10
t_plot = t[::downsample_factor]
phi_initial_plot = phi_initial[::downsample_factor]
phi_evolved_plot = phi_evolved[::downsample_factor]
s_initial_plot = s_initial[::downsample_factor]
s_evolved_plot = s_evolved[::downsample_factor]

# Plotting
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t_plot, phi_initial_plot, label=r'Initial $\phi(t)$')
plt.plot(t_plot, phi_evolved_plot, label=r'Evolved $\phi(t)$')
plt.xlabel('Time $t$')
plt.ylabel(r'$\phi(t)$')
plt.title(r'Evolution of the Perturbation $\phi(t)$')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(t_plot, s_initial_plot, label=r'Initial Schwarzian $\{f(t), t\}$')
plt.plot(t_plot, s_evolved_plot, label=r'Evolved Schwarzian $\{f(t), t\}$')
plt.xlabel('Time $t$')
plt.ylabel('Schwarzian Derivative')
plt.title('Schwarzian Derivative Before and After Evolution')
plt.legend()

plt.tight_layout()
plt.show()
