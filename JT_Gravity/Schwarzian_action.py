import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp
from scipy.interpolate import interp1d

# Constants
C = 1.0        # Coupling constant
S0 = -0.1      # Schwarzian derivative constant (set to a negative value for non-zero energy)

# Time parameters
t_start = 0.0
t_end = 10.0
N = 10000
t = np.linspace(t_start, t_end, N)

# Initial perturbation parameters
delta = 0.01   # Increased perturbation amplitude
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

# Equations of motion derived from the Schwarzian action
def equations_of_motion(t, y):
    """
    Defines the system of ODEs representing the equations of motion derived from the Schwarzian action.
    Parameters:
        t (float): Time variable.
        y (list): List containing [f(t), f'(t), f''(t)].
    Returns:
        list: Derivatives [f'(t), f''(t), f'''(t)].
    """
    y0, y1, y2 = y  # y0 = f(t), y1 = f'(t), y2 = f''(t)
    epsilon = 1e-12
    # Regularize y1 to prevent division by zero
    y1_safe = y1 if np.abs(y1) > epsilon else epsilon * np.sign(y1)
    # Compute f'''(t) based on the equations of motion
    y3 = S0 * y1_safe + (1.5) * (y2 ** 2) / y1_safe
    return [y1, y2, y3]

# Solve the ODE system using adaptive time stepping
sol = solve_ivp(
    equations_of_motion,
    [t_start, t_end],
    y0,
    method='DOP853',   # Non-stiff solver with adaptive time stepping
    atol=1e-10,
    rtol=1e-10
)

# Interpolate the solution to the desired time points
f_evolved_interp = interp1d(sol.t, sol.y[0], kind='cubic', fill_value="extrapolate")
f_evolved = f_evolved_interp(t)
phi_evolved = f_evolved - t

# Compute Schwarzian derivative with adaptive regularization
def schwarzian_derivative(f, t):
    """
    Computes the Schwarzian derivative of f with respect to t.
    Parameters:
        f (array): Function values.
        t (array): Time values.
    Returns:
        array: Schwarzian derivative values.
    """
    # Use higher-order finite differences for better accuracy
    f_prime = np.gradient(f, t, edge_order=2)
    f_double_prime = np.gradient(f_prime, t, edge_order=2)
    f_triple_prime = np.gradient(f_double_prime, t, edge_order=2)
    epsilon = 1e-8
    f_prime_safe = np.where(np.abs(f_prime) < epsilon, epsilon * np.sign(f_prime), f_prime)
    s = (f_triple_prime / f_prime_safe) - 1.5 * (f_double_prime / f_prime_safe) ** 2
    return s

s_initial = schwarzian_derivative(t + phi_initial, t)
s_evolved = schwarzian_derivative(f_evolved, t)

# Compute energy from the Schwarzian derivative
energy_initial = -s_initial / (2 * np.pi**2)
energy_evolved = -s_evolved / (2 * np.pi**2)

# Plotting
plt.figure(figsize=(12, 12))

plt.subplot(3, 1, 1)
plt.plot(t, phi_initial, label=r'Initial $\phi(t)$')
plt.plot(t, phi_evolved, label=r'Evolved $\phi(t)$')
plt.xlabel('Time $t$')
plt.ylabel(r'$\phi(t)$')
plt.title(r'Evolution of the Perturbation $\phi(t)$')
plt.legend()

plt.subplot(3, 1, 2)
plt.plot(t, s_initial, label=r'Initial Schwarzian $\{f(t), t\}$')
plt.plot(t, s_evolved, label=r'Evolved Schwarzian $\{f(t), t\}$')
plt.xlabel('Time $t$')
plt.ylabel('Schwarzian Derivative')
plt.title('Schwarzian Derivative Before and After Evolution')
plt.legend()

plt.subplot(3, 1, 3)
plt.plot(t, energy_evolved, label='Evolved Energy')
plt.xlabel('Time $t$')
plt.ylabel('Energy $E$')
plt.title('Energy Evolution')
plt.legend()

plt.tight_layout()
plt.show()
