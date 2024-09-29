import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp

# Constants
C = 1.0  # Coupling constant
S0 = 0.0  # Schwarzian derivative constant

# Time parameters
t_start = 0.0
t_end = 10.0
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

# Create numerical functions from symbolic expressions
phi_func = sp.lambdify(t_sym, phi_sym, modules='numpy')
phi_prime_func = sp.lambdify(t_sym, phi_prime_sym, modules='numpy')
phi_double_prime_func = sp.lambdify(t_sym, phi_double_prime_sym, modules='numpy')

# Evaluate the functions
phi_initial = phi_func(t)
phi_prime = phi_prime_func(t)
phi_double_prime = phi_double_prime_func(t)

# Initial conditions
f0 = t[0] + phi_initial[0]
f_prime0 = 1 + phi_prime[0]
f_double_prime0 = phi_double_prime[0]
y0 = [f0, f_prime0, f_double_prime0]

# Equations of motion derived from Schwarzian action
def equations_of_motion(t, y):
    y0, y1, y2 = y  # y0 = f(t), y1 = f'(t), y2 = f''(t)
    # Avoid division by zero
    if np.abs(y1) < 1e-12:
        y1 = np.sign(y1) * 1e-12
    y3 = S0 * y1 + (1.5) * (y2 ** 2) / y1
    return [y1, y2, y3]

# Solve the ODE system using a stiff solver
sol = solve_ivp(
    equations_of_motion,
    [t_start, t_end],
    y0,
    t_eval=t,
    method='Radau',  # Stiff solver
    atol=1e-8,
    rtol=1e-8
)

# Extract the solution
f_evolved = sol.y[0]
phi_evolved = f_evolved - t

# Compute Schwarzian derivative with adaptive regularization
def schwarzian_derivative(f, t):
    f_prime = np.gradient(f, t)
    f_double_prime = np.gradient(f_prime, t)
    f_triple_prime = np.gradient(f_double_prime, t)
    epsilon = 1e-8 * np.max(np.abs(f_prime))
    f_prime_safe = np.where(np.abs(f_prime) < epsilon, epsilon, f_prime)
    s = (f_triple_prime / f_prime_safe) - (1.5) * (f_double_prime / f_prime_safe) ** 2
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

