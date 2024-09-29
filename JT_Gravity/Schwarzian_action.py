import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import sympy as sp

# Constants
C = 1.0          # Coupling constant (dimensionless)
delta = 0.01     # Perturbation amplitude
t_start = 0.0
t_end = 10.0
N = 10000
epsilon = 1e-8   # Small value to prevent division by zero

def main():
    # Time array
    t_np = np.linspace(t_start, t_end, N)
    
    # Compute initial perturbations and their derivatives
    phi_initial, phi_prime, phi_double_prime, phi_triple_prime, \
    dilaton_initial, dilaton_prime_initial = compute_initial_perturbations(t_np, delta)
    
    # Compute initial Schwarzian derivative and related functions
    f_initial, f_prime_initial, f_double_prime_initial, \
    S_initial, S0 = compute_initial_schwarzian(t_np, phi_initial, phi_prime, phi_double_prime, phi_triple_prime, epsilon)
    
    # Set initial conditions for the ODE solver
    initial_conditions = [
        f_initial[0],
        f_prime_initial[0],
        f_double_prime_initial[0],
        dilaton_initial[0],
        dilaton_prime_initial[0]
    ]
    
    # Solve the equations of motion
    sol = solve_equations_of_motion(t_np, initial_conditions, S0, C, epsilon)
    
    # Process the solution and plot the results
    process_and_plot_solution(t_np, sol, phi_initial, S_initial, f_initial, epsilon, S0)

def compute_initial_perturbations(t_np, delta):
    """
    Computes the initial perturbation functions and their derivatives.

    Parameters:
    t_np (numpy.ndarray): Array of time points.
    delta (float): Perturbation amplitude.

    Returns:
    tuple: Contains phi_initial, phi_prime, phi_double_prime, phi_triple_prime,
           dilaton_initial, dilaton_prime_initial.
    """
    # Initialize symbolic variables
    t_sym = sp.symbols('t')
    delta_sym = sp.Float(delta)
    omega = 2 * sp.pi / (t_np[-1] - t_np[0])
    
    # Define perturbation and dilaton symbolically
    phi_sym = delta_sym * sp.sin(omega * t_sym)
    dilaton_sym = delta_sym * sp.cos(omega * t_sym)
    
    # Compute symbolic derivatives
    phi_prime_sym = sp.diff(phi_sym, t_sym)
    phi_double_prime_sym = sp.diff(phi_prime_sym, t_sym)
    phi_triple_prime_sym = sp.diff(phi_double_prime_sym, t_sym)
    
    dilaton_prime_sym = sp.diff(dilaton_sym, t_sym)
    
    # Create numerical functions
    phi_func = sp.lambdify(t_sym, phi_sym, modules='numpy')
    phi_prime_func = sp.lambdify(t_sym, phi_prime_sym, modules='numpy')
    phi_double_prime_func = sp.lambdify(t_sym, phi_double_prime_sym, modules='numpy')
    phi_triple_prime_func = sp.lambdify(t_sym, phi_triple_prime_sym, modules='numpy')
    dilaton_func = sp.lambdify(t_sym, dilaton_sym, modules='numpy')
    dilaton_prime_func = sp.lambdify(t_sym, dilaton_prime_sym, modules='numpy')
    
    # Evaluate the functions
    phi_initial = phi_func(t_np)
    phi_prime = phi_prime_func(t_np)
    phi_double_prime = phi_double_prime_func(t_np)
    phi_triple_prime = phi_triple_prime_func(t_np)
    dilaton_initial = dilaton_func(t_np)
    dilaton_prime_initial = dilaton_prime_func(t_np)
    
    return phi_initial, phi_prime, phi_double_prime, phi_triple_prime, dilaton_initial, dilaton_prime_initial

def compute_initial_schwarzian(t_np, phi_initial, phi_prime, phi_double_prime, phi_triple_prime, epsilon):
    """
    Computes the initial Schwarzian derivative and related functions.

    Parameters:
    t_np (numpy.ndarray): Array of time points.
    phi_initial (numpy.ndarray): Initial phi(t).
    phi_prime (numpy.ndarray): First derivative of phi(t).
    phi_double_prime (numpy.ndarray): Second derivative of phi(t).
    phi_triple_prime (numpy.ndarray): Third derivative of phi(t).
    epsilon (float): Small value to prevent division by zero.

    Returns:
    tuple: Contains f_initial, f_prime_initial, f_double_prime_initial,
           S_initial, S0.
    """
    # Compute f and its derivatives
    f_initial = t_np + phi_initial
    f_prime_initial = 1 + phi_prime
    f_double_prime_initial = phi_double_prime
    f_triple_prime_initial = phi_triple_prime
    
    # Compute initial Schwarzian derivative
    f_prime_safe = np.where(np.abs(f_prime_initial) > epsilon, f_prime_initial, epsilon)
    S_initial = (f_triple_prime_initial / f_prime_safe) - 1.5 * (f_double_prime_initial / f_prime_safe) ** 2
    
    # Set Schwarzian derivative constant
    S0 = np.mean(S_initial)
    
    return f_initial, f_prime_initial, f_double_prime_initial, S_initial, S0

def solve_equations_of_motion(t_np, initial_conditions, S0, C, epsilon):
    """
    Solves the equations of motion for the system.

    Parameters:
    t_np (numpy.ndarray): Array of time points.
    initial_conditions (list): Initial conditions for the ODE solver.
    S0 (float): Initial Schwarzian derivative constant.
    C (float): Coupling constant.
    epsilon (float): Small value to prevent division by zero.

    Returns:
    OdeResult: Solution object from solve_ivp.
    """
    def equations_of_motion(t, y):
        f, f_prime, f_double_prime, phi, phi_prime = y
    
        f_prime_safe = f_prime if np.abs(f_prime) > epsilon else epsilon
        f_triple_prime = f_prime_safe * (S0 + 1.5 * (f_double_prime / f_prime_safe) ** 2) + C * phi_prime
        phi_double_prime = -C * f_prime
    
        return [f_prime, f_double_prime, f_triple_prime, phi_prime, phi_double_prime]
    
    # Solve the ODE system
    sol = solve_ivp(
        equations_of_motion,
        [t_np[0], t_np[-1]],
        initial_conditions,
        method='Radau',
        t_eval=t_np,
        atol=1e-6,
        rtol=1e-6,
        max_step=0.1
    )
    
    if not sol.success:
        print("Solver failed:", sol.message)
        exit()
    
    return sol

def process_and_plot_solution(t_np, sol, phi_initial, S_initial, f_initial, epsilon, S0):
    """
    Processes the solution and plots the results.

    Parameters:
    t_np (numpy.ndarray): Array of time points.
    sol (OdeResult): Solution object from solve_ivp.
    phi_initial (numpy.ndarray): Initial phi(t).
    S_initial (numpy.ndarray): Initial Schwarzian derivative.
    f_initial (numpy.ndarray): Initial f(t).
    epsilon (float): Small value to prevent division by zero.
    S0 (float): Initial Schwarzian derivative constant.
    """
    # Extract the solution
    f_evolved = sol.y[0]
    f_prime_evolved = sol.y[1]
    f_double_prime_evolved = sol.y[2]
    phi_evolved = sol.y[3]
    phi_prime_evolved = sol.y[4]
    
    # Compute the evolved Schwarzian derivative
    f_prime_evolved_safe = np.where(np.abs(f_prime_evolved) > epsilon, f_prime_evolved, epsilon)
    f_triple_prime_evolved = (
        f_prime_evolved_safe * (S0 + 1.5 * (f_double_prime_evolved / f_prime_evolved_safe) ** 2) 
        + C * phi_prime_evolved
    )
    S_evolved = (f_triple_prime_evolved / f_prime_evolved_safe) - 1.5 * (f_double_prime_evolved / f_prime_evolved_safe) ** 2
    phi_evolved_from_f = f_evolved - t_np
    
    # Plotting
    plt.figure(figsize=(12, 20))
    
    plt.subplot(5, 1, 1)
    plt.plot(t_np, phi_initial, label=r'Initial $\phi(t)$')
    plt.plot(t_np, phi_evolved_from_f, label=r'Evolved $\phi(t)$')
    plt.xlabel('Time $t$')
    plt.ylabel(r'$\phi(t)$')
    plt.title(r'Evolution of the Perturbation $\phi(t)$')
    plt.legend()
    
    plt.subplot(5, 1, 2)
    plt.plot(t_np, S_initial, label=r'Initial Schwarzian $\{f(t), t\}$')
    plt.plot(t_np, S_evolved, label=r'Evolved Schwarzian $\{f(t), t\}$')
    plt.xlabel('Time $t$')
    plt.ylabel('Schwarzian Derivative')
    plt.title('Schwarzian Derivative Before and After Evolution')
    plt.legend()
    
    plt.subplot(5, 1, 3)
    plt.plot(t_np, f_evolved - f_initial, label='Difference between Evolved and Initial $f(t)$')
    plt.xlabel('Time $t$')
    plt.ylabel('Difference')
    plt.title('Difference between Evolved and Initial Functions')
    plt.legend()
    
    plt.subplot(5, 1, 4)
    plt.plot(t_np, f_prime_evolved, label=r'Evolved $f\'(t)$')
    plt.xlabel('Time $t$')
    plt.ylabel(r'$f\'(t)$')
    plt.title('First Derivative of Evolved Function')
    plt.legend()
    
    plt.subplot(5, 1, 5)
    plt.plot(t_np, phi_evolved, label=r'Evolved Dilaton $\phi(t)$')
    plt.xlabel('Time $t$')
    plt.ylabel(r'$\phi(t)$')
    plt.title('Evolution of the Dilaton Field')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
