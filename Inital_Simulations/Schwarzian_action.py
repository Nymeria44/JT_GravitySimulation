import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
import sympy as sp
from dataclasses import dataclass

# Configuration using dataclass
@dataclass
class Config:
    C: float = 1.0
    delta: float = 0.01
    t_start: float = 0.0
    t_end: float = 3.0
    N: int = 1_000_000
    epsilon: float = float(np.finfo(np.float64).eps)  # Machine epsilon
    delta_T: float = 0.01
    omega_T: float = 0.5

config = Config()

# Data classes for structured data management
@dataclass
class SymbolicPerturbations:
    phi: sp.Expr
    phi_prime: sp.Expr
    phi_double_prime: sp.Expr
    phi_triple_prime: sp.Expr
    dilaton: sp.Expr
    dilaton_prime: sp.Expr

@dataclass
class NumericalPerturbations:
    phi: np.ndarray
    phi_prime: np.ndarray
    phi_double_prime: np.ndarray
    phi_triple_prime: np.ndarray
    dilaton: np.ndarray
    dilaton_prime: np.ndarray

@dataclass
class SchwarzianData:
    f: np.ndarray
    f_prime: np.ndarray
    f_double_prime: np.ndarray
    f_triple_prime: np.ndarray
    S: np.ndarray

def main():
    """Main computation function."""

    # Generate time array
    t_np = np.linspace(config.t_start, config.t_end, config.N)
    t_range = config.t_end - config.t_start

    # Compute symbolic initial perturbations and derivatives
    symbolic_perturbations = define_symbolic_perturbations(config.delta, t_range)

    # Convert symbolic perturbations to numerical functions
    numerical_perturbations = evaluate_perturbations(t_np, symbolic_perturbations)

    # Compute initial Schwarzian derivative
    initial_schwarzian = compute_initial_schwarzian(t_np, numerical_perturbations)

    # Interpolate S_initial to create a continuous function S0(t)
    S0_spline = CubicSpline(t_np, initial_schwarzian.S, bc_type='natural')

    # Set initial conditions for the ODE solver
    initial_conditions = [
        initial_schwarzian.f[0],
        initial_schwarzian.f_prime[0],
        initial_schwarzian.f_double_prime[0],
        numerical_perturbations.dilaton[0],
        numerical_perturbations.dilaton_prime[0]
    ]

    # Solve the equations of motion
    sol = solve_equations_of_motion(t_np, initial_conditions, S0_spline)

    # Process the solution and plot the results
    process_and_plot_solution(t_np, sol, numerical_perturbations.phi, initial_schwarzian.S, initial_schwarzian.f, S0_spline)

def temperature_perturbation(t: np.ndarray, delta_T: float, omega_T: float) -> np.ndarray:
    """Defines the time-dependent temperature perturbation."""
    return 1.0 + delta_T * np.sin(omega_T * t)

def define_symbolic_perturbations(delta: float, t_range: float) -> SymbolicPerturbations:
    """Defines symbolic expressions for perturbations and their derivatives."""

    t_sym = sp.symbols('t')
    omega = 2 * sp.pi / t_range

    delta_sym = sp.Float(delta)
    phi_sym = delta_sym * sp.sin(omega * t_sym)
    dilaton_sym = delta_sym * sp.cos(omega * t_sym)

    phi_prime_sym = sp.diff(phi_sym, t_sym)
    phi_double_prime_sym = sp.diff(phi_prime_sym, t_sym)
    phi_triple_prime_sym = sp.diff(phi_double_prime_sym, t_sym)
    dilaton_prime_sym = sp.diff(dilaton_sym, t_sym)

    return SymbolicPerturbations(
        phi=phi_sym,
        phi_prime=phi_prime_sym,
        phi_double_prime=phi_double_prime_sym,
        phi_triple_prime=phi_triple_prime_sym,
        dilaton=dilaton_sym,
        dilaton_prime=dilaton_prime_sym
    )

def evaluate_perturbations(t_np: np.ndarray, symbolic_perturbations: SymbolicPerturbations) -> NumericalPerturbations:
    """Evaluates symbolic perturbations over the time array to obtain numerical perturbations."""

    t_sym = sp.symbols('t')
    numerical_values = {}

    for name in symbolic_perturbations.__dataclass_fields__:
        expr = getattr(symbolic_perturbations, name)
        func = sp.lambdify(t_sym, expr, modules='numpy')
        numerical_values[name] = func(t_np)

    return NumericalPerturbations(**numerical_values)

def compute_initial_schwarzian(t_np: np.ndarray, numerical_perturbations: NumericalPerturbations) -> SchwarzianData:
    """Computes the initial Schwarzian derivative and related data."""

    phi = numerical_perturbations.phi
    phi_prime = numerical_perturbations.phi_prime
    phi_double_prime = numerical_perturbations.phi_double_prime
    phi_triple_prime = numerical_perturbations.phi_triple_prime

    f = t_np + phi
    f_prime = 1 + phi_prime
    f_double_prime = phi_double_prime
    f_triple_prime = phi_triple_prime

    f_prime_safe = np.where(f_prime > config.epsilon, f_prime, config.epsilon)
    ratio = f_double_prime / f_prime_safe
    S = (f_triple_prime / f_prime_safe) - 1.5 * ratio**2

    return SchwarzianData(
        f=f,
        f_prime=f_prime,
        f_double_prime=f_double_prime,
        f_triple_prime=f_triple_prime,
        S=S
    )

def solve_equations_of_motion(t_np: np.ndarray, initial_conditions, S0_spline):
    """Solves the equations of motion using the initial conditions and the interpolated Schwarzian derivative."""

    def equations_of_motion(t, y):
        f, f_prime, f_double_prime, phi, phi_prime = y

        temperature = temperature_perturbation(t, config.delta_T, config.omega_T)
        S0_t = S0_spline(t)
        S_time_dependent = S0_t * temperature

        f_prime_safe = max(f_prime, config.epsilon)
        ratio = f_double_prime / f_prime_safe

        f_triple_prime = f_prime_safe * (S_time_dependent + 1.5 * ratio**2) + config.C * phi_prime
        phi_double_prime = -config.C * f_prime

        return [f_prime, f_double_prime, f_triple_prime, phi_prime, phi_double_prime]

    sol = solve_ivp(
        equations_of_motion,
        [t_np[0], t_np[-1]],
        initial_conditions,
        method='Radau',
        t_eval=t_np,
        atol=1e-8,
        rtol=1e-8
    )

    if not sol.success:
        raise RuntimeError(f"Solver failed: {sol.message}")

    return sol

def process_and_plot_solution(t_np, sol, phi_initial, S_initial, f_initial, S0_spline):
    """Processes the solution from the ODE solver and plots the results."""

    f_evolved, f_prime_evolved, f_double_prime_evolved, phi_evolved, phi_prime_evolved = sol.y

    f_prime_evolved_safe = np.where(f_prime_evolved > config.epsilon, f_prime_evolved, config.epsilon)
    ratio = f_double_prime_evolved / f_prime_evolved_safe

    temperature = temperature_perturbation(t_np, config.delta_T, config.omega_T)
    S0_t = S0_spline(t_np)
    S_time_dependent = S0_t * temperature

    f_triple_prime_evolved = f_prime_evolved_safe * (S_time_dependent + 1.5 * ratio**2) + config.C * phi_prime_evolved
    S_evolved = (f_triple_prime_evolved / f_prime_evolved_safe) - 1.5 * ratio**2
    phi_evolved_from_f = f_evolved - t_np

    plot_solutions(
        t_np,
        phi_initial,
        phi_evolved_from_f,
        S_initial,
        S_evolved,
        temperature,
        f_evolved,
        f_prime_evolved,
        phi_evolved,
        f_initial
    )

def plot_solutions(t_np, phi_initial, phi_evolved_from_f, S_initial, S_evolved, temperature, f_evolved, f_prime_evolved, phi_evolved, f_initial):
    """Plots the initial and evolved solutions and their differences."""

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
