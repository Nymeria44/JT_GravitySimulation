import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import sympy as sp
from dataclasses import dataclass

# Configuration Section using dataclass
@dataclass
class Config:
    C: float = 1.0
    delta: float = 0.01
    t_start: float = 0.0
    t_end: float = 3.0
    N: int = 1000000
    epsilon: float = float(np.finfo(np.float64).eps)  # Convert to Python float
    delta_T: float = 0.01
    omega_T: float = 0.5

config = Config()

def main():
    print("Starting main computation...")
    
    # Time array
    t_np = np.linspace(config.t_start, config.t_end, config.N)
    print(f"Generated time array with {len(t_np)} points from {config.t_start} to {config.t_end}")

    # Compute initial perturbations and derivatives with symbolic differentiation
    print("Computing initial perturbations...")
    phi_initial, phi_prime, phi_double_prime, phi_triple_prime, dilaton_initial, dilaton_prime_initial = compute_initial_perturbations(t_np, config.delta)
    print("Initial perturbations computed.")

    # Compute initial Schwarzian derivative
    print("Computing initial Schwarzian derivative...")
    f_initial, f_prime_initial, f_double_prime_initial, S_initial = compute_initial_schwarzian(
        t_np, phi_initial, phi_prime, phi_double_prime, phi_triple_prime
    )
    print("Initial Schwarzian derivative computed.")

    # Interpolate S_initial to create a continuous function S0(t) using CubicSpline with boundary condition
    print("Interpolating initial Schwarzian derivative...")
    S0_spline = CubicSpline(t_np, S_initial, bc_type='natural')
    print("Interpolation completed.")

    # Set initial conditions for the ODE solver
    initial_conditions = [f_initial[0], f_prime_initial[0], f_double_prime_initial[0], dilaton_initial[0], dilaton_prime_initial[0]]
    print(f"Initial conditions set: {initial_conditions}")

    # Solve the equations of motion
    print("Solving equations of motion...")
    sol = solve_equations_of_motion(t_np, initial_conditions, S0_spline)
    print("Equations of motion solved.")

    # Process the solution and plot the results
    print("Processing and plotting solution...")
    process_and_plot_solution(t_np, sol, phi_initial, S_initial, f_initial, S0_spline)
    print("Solution processed and plotted.")

def temperature_perturbation(t: np.ndarray, delta_T: float, omega_T: float) -> np.ndarray:
    return 1.0 + delta_T * np.sin(omega_T * t)

def compute_initial_perturbations(t_np: np.ndarray, delta: float):
    t_sym = sp.symbols('t')
    omega = 2 * sp.pi / (t_np[-1] - t_np[0])

    phi_sym = delta * sp.sin(omega * t_sym)
    dilaton_sym = delta * sp.cos(omega * t_sym)

    phi_prime_sym = sp.diff(phi_sym, t_sym)
    phi_double_prime_sym = sp.diff(phi_prime_sym, t_sym)
    phi_triple_prime_sym = sp.diff(phi_double_prime_sym, t_sym)
    dilaton_prime_sym = sp.diff(dilaton_sym, t_sym)

    phi_func = sp.lambdify(t_sym, phi_sym, modules='numpy')
    phi_prime_func = sp.lambdify(t_sym, phi_prime_sym, modules='numpy')
    phi_double_prime_func = sp.lambdify(t_sym, phi_double_prime_sym, modules='numpy')
    phi_triple_prime_func = sp.lambdify(t_sym, phi_triple_prime_sym, modules='numpy')
    dilaton_func = sp.lambdify(t_sym, dilaton_sym, modules='numpy')
    dilaton_prime_func = sp.lambdify(t_sym, dilaton_prime_sym, modules='numpy')

    phi_initial = phi_func(t_np)
    phi_prime = phi_prime_func(t_np)
    phi_double_prime = phi_double_prime_func(t_np)
    phi_triple_prime = phi_triple_prime_func(t_np)
    dilaton_initial = dilaton_func(t_np)
    dilaton_prime_initial = dilaton_prime_func(t_np)

    print("Initial perturbations and their derivatives computed.")

    return phi_initial, phi_prime, phi_double_prime, phi_triple_prime, dilaton_initial, dilaton_prime_initial

def compute_initial_schwarzian(t_np: np.ndarray, phi_initial: np.ndarray, phi_prime: np.ndarray, phi_double_prime: np.ndarray, phi_triple_prime: np.ndarray):
    f_initial = t_np + phi_initial
    f_prime_initial = 1 + phi_prime
    f_double_prime_initial = phi_double_prime
    f_triple_prime_initial = phi_triple_prime

    f_prime_safe = np.clip(f_prime_initial, config.epsilon, None)
    ratio = f_double_prime_initial / f_prime_safe
    S_initial = (f_triple_prime_initial / f_prime_safe) - 1.5 * ratio**2

    print("Initial Schwarzian derivative computed.")

    return f_initial, f_prime_initial, f_double_prime_initial, S_initial

def solve_equations_of_motion(t_np: np.ndarray, initial_conditions, S0_spline):
    def equations_of_motion(t, y):
        f, f_prime, f_double_prime, phi, phi_prime = y
        temperature = temperature_perturbation(t, config.delta_T, config.omega_T)

        S0_t = S0_spline(t)
        S_time_dependent = S0_t * temperature

        f_prime_safe = np.clip(f_prime, config.epsilon, None)
        ratio = f_double_prime / f_prime_safe
        f_triple_prime = f_prime_safe * (S_time_dependent + 1.5 * ratio**2) + config.C * phi_prime
        phi_double_prime = -config.C * f_prime

        return [f_prime, f_double_prime, f_triple_prime, phi_prime, phi_double_prime]

    print("Starting ODE solver...")
    sol = solve_ivp(equations_of_motion, [t_np[0], t_np[-1]], initial_conditions, method='Radau', t_eval=t_np, atol=1e-8, rtol=1e-8)

    if not sol.success:
        print(f"Solver failed: {sol.message}")
        raise RuntimeError(f"Solver failed: {sol.message}. Consider adjusting solver parameters or reviewing initial conditions.")
    else:
        print("ODE solver completed successfully.")

    return sol

def process_and_plot_solution(t_np, sol, phi_initial, S_initial, f_initial, S0_spline):
    f_evolved = sol.y[0]
    f_prime_evolved = sol.y[1]
    f_double_prime_evolved = sol.y[2]
    phi_evolved = sol.y[3]
    phi_prime_evolved = sol.y[4]

    print("Processing solution...")

    f_prime_evolved_safe = np.where(np.abs(f_prime_evolved) > config.epsilon, f_prime_evolved, np.sign(f_prime_evolved) * config.epsilon)
    ratio = f_double_prime_evolved / f_prime_evolved_safe

    temperature = temperature_perturbation(t_np, config.delta_T, config.omega_T)
    S0_t = S0_spline(t_np)
    S_time_dependent = S0_t * temperature

    f_triple_prime_evolved = f_prime_evolved_safe * (S_time_dependent + 1.5 * ratio ** 2) + config.C * phi_prime_evolved
    S_evolved = (f_triple_prime_evolved / f_prime_evolved_safe) - 1.5 * ratio ** 2
    phi_evolved_from_f = f_evolved - t_np

    print("Solution processed, now plotting...")
    plot_solutions(t_np, phi_initial, phi_evolved_from_f, S_initial, S_evolved, temperature, f_evolved, f_prime_evolved, phi_evolved, f_initial)
    print("Plotting completed.")

def plot_solutions(t_np, phi_initial, phi_evolved_from_f, S_initial, S_evolved, temperature, f_evolved, f_prime_evolved, phi_evolved, f_initial):
    plt.figure(figsize=(12, 24))

    plt.subplot(6, 1, 1)
    plt.plot(t_np, phi_initial, label=r'Initial $\phi(t)$')
    plt.plot(t_np, phi_evolved_from_f, label=r'Evolved $\phi(t)$')
    plt.xlabel('Time $t$')
    plt.ylabel(r'$\phi(t)$')
    plt.title(r'Evolution of the Perturbation $\phi(t)$')
    plt.legend()

    plt.subplot(6, 1, 2)
    plt.plot(t_np, S_initial, label=r'Initial Schwarzian $\{f(t), t\}$')
    plt.plot(t_np, S_evolved, label=r'Evolved Schwarzian $\{f(t), t\}$')
    plt.xlabel('Time $t$')
    plt.ylabel('Schwarzian Derivative')
    plt.title('Schwarzian Derivative Before and After Evolution')
    plt.legend()

    plt.subplot(6, 1, 3)
    plt.plot(t_np, temperature, label=r'Temperature Perturbation $T(t)$')
    plt.xlabel('Time $t$')
    plt.ylabel('Temperature')
    plt.title('Time-Dependent Temperature Perturbation')
    plt.legend()

    plt.subplot(6, 1, 4)
    plt.plot(t_np, f_evolved - f_initial, label='Difference between Evolved and Initial $f(t)$')
    plt.xlabel('Time $t$')
    plt.ylabel('Difference')
    plt.title('Difference between Evolved and Initial Functions')
    plt.legend()

    plt.subplot(6, 1, 5)
    plt.plot(t_np, f_prime_evolved, label=r'Evolved $f\'(t)$')
    plt.xlabel('Time $t$')
    plt.ylabel(r'$f\'(t)$')
    plt.title('First Derivative of Evolved Function')
    plt.legend()

    plt.subplot(6, 1, 6)
    plt.plot(t_np, phi_evolved, label=r'Evolved Dilaton $\phi(t)$')
    plt.xlabel('Time $t$')
    plt.ylabel(r'$\phi(t)$')
    plt.title('Evolution of the Dilaton Field')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
