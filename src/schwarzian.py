# schwarzianJAX.py

import jax
import jax.numpy as jnp

from optimisations import (
    run_bfgs_optimization,
    run_adam_optimization,
    run_optax_adam_optimization,
    run_newtons_method,
    run_hessian_optimization,
    run_yogi_optimization,
    run_lbfgs_optimization,
    run_adabelief_optimization,
    print_final_comparison
)

from config import PerturbationConfig  # Import the configuration class

################################################################################
# Fourier Series and Gaussian Pulse Calculations
################################################################################

def calculate_delta_g(t, p, M, T, n, order=0, pulse_time=None, pulse_amp=0, pulse_width=0.00):
    """
    Calculate Fourier series terms and Gaussian pulse for g(t).

    Parameters
    ----------
    t : jnp.ndarray
        Time points for evaluation
    p : jnp.ndarray
        Fourier coefficients (sin and cos combined)
    M : int
        Number of terms in each series
    T : float
        Period
    n : jnp.ndarray
        Harmonic indices
    order : int
        Derivative order (0=function, 1=first derivative, etc.)
    pulse_time : float, optional
        Gaussian pulse center time
    pulse_amp : float, optional
        Gaussian pulse amplitude
    pulse_width : float, optional
        Gaussian pulse width

    Returns
    -------
    jnp.ndarray
        Fourier series perturbation or its derivative
    """   
    delta_g = 0.0  # Initialized to zero

    if p is not None and p.size > 0:
        # Separate sine and cosine coefficients
        sin_coeffs = p[:M]
        cos_coeffs = p[M:]
        
        # Calculate frequency factor based on derivative order
        factor = (2 * jnp.pi * n / T) ** order
        
        # Calculate sine and cosine terms, adjusting for derivative order
        if order % 4 == 0:  # No change in sin/cos phase
            sin_terms = jnp.sin(2 * jnp.pi * n[:, None] * t[None, :] / T)
            cos_terms = jnp.cos(2 * jnp.pi * n[:, None] * t[None, :] / T)
        elif order % 4 == 1:  # First derivative
            sin_terms = jnp.cos(2 * jnp.pi * n[:, None] * t[None, :] / T)
            cos_terms = -jnp.sin(2 * jnp.pi * n[:, None] * t[None, :] / T)
        elif order % 4 == 2:  # Second derivative
            sin_terms = -jnp.sin(2 * jnp.pi * n[:, None] * t[None, :] / T)
            cos_terms = -jnp.cos(2 * jnp.pi * n[:, None] * t[None, :] / T)
        else:  # Third derivative
            sin_terms = -jnp.cos(2 * jnp.pi * n[:, None] * t[None, :] / T)
            cos_terms = jnp.sin(2 * jnp.pi * n[:, None] * t[None, :] / T)

        # Apply factor for each order to coefficients and calculate delta_g
        delta_g = jnp.dot(sin_coeffs * factor, sin_terms) + jnp.dot(cos_coeffs * factor, cos_terms)
    
    # Optionally add a Gaussian pulse and its derivatives if pulse_time is specified
    if pulse_time is not None and pulse_amp != 0 and pulse_width != 0:
        # Gaussian pulse and its derivatives
        t_diff = t - pulse_time
        gaussian_base = pulse_amp / jnp.sqrt(2 * jnp.pi * pulse_width**2)
        
        if order == 0:
            # 0th derivative: Gaussian pulse itself
            gaussian_pulse = gaussian_base * jnp.exp(-t_diff**2 / (2 * pulse_width**2))
        elif order == 1:
            # 1st derivative
            gaussian_pulse = gaussian_base * (-t_diff / pulse_width**2) * jnp.exp(-t_diff**2 / (2 * pulse_width**2))
        elif order == 2:
            # 2nd derivative
            gaussian_pulse = gaussian_base * ((t_diff**2 - pulse_width**2) / pulse_width**4) * jnp.exp(-t_diff**2 / (2 * pulse_width**2))
        elif order == 3:
            # 3rd derivative
            gaussian_pulse = gaussian_base * (-t_diff * (t_diff**2 - 3 * pulse_width**2) / pulse_width**6) * jnp.exp(-t_diff**2 / (2 * pulse_width**2))
        else:
            # Higher derivatives are not implemented here
            raise ValueError("Higher derivatives are not implemented for the Gaussian pulse")

        # Add the Gaussian pulse or its derivative to delta_g
        delta_g += gaussian_pulse

    return delta_g

################################################################################
# Core Function Calculations (f(t) and derivatives)
################################################################################

def calculate_f(p_opt, config: PerturbationConfig, order=0):
    """
    Calculate f(t) and its derivatives using log-dampened approach.

    Parameters
    ----------
    p_opt : jnp.ndarray
        Optimizer-controlled Fourier coefficients
    config : PerturbationConfig
        Configuration containing grid and parameters
    order : int, default=0
        Derivative order (0=f, 1=f', 2=f'', 3=f''')

    Returns
    -------
    jnp.ndarray
        Function or derivative values on time grid

    Raises
    ------
    ValueError
        If order > 3
    """
    # Compute delta_g_user and delta_g_opt (keep existing code)
    delta_g_user = calculate_delta_g(
        t=config._t, p=config.p_user,
        M=config.M_user, T=config.T,
        n=config.n_user, order=order,
        pulse_time=config.pulse_time,
        pulse_amp=config.pulse_amp,
        pulse_width=config.pulse_width
    )

    delta_g_opt = calculate_delta_g(
        t=config._t, p=p_opt,
        M=config.M_opt, T=config.T,
        n=config.n_opt, order=order
    )

    # Total g(t) or its derivatives
    g_t = delta_g_user + delta_g_opt

    if order == 0:
        # Use log1p(exp(x)) for numerical stability
        # This grows more slowly than exp(x) while maintaining monotonicity
        dampened_exp = jnp.log1p(jnp.exp(g_t))
        
        # Compute the integral using cumulative sum
        integral_dampened = jnp.cumsum(dampened_exp) * config.dt
        f_t = config._t + integral_dampened
        return f_t
        
    elif order == 1:
        # First derivative: f'(t) = 1 + exp(g(t))/(1 + exp(g(t)))
        # This is the derivative of log1p(exp(x))
        f_prime = 1 + jnp.exp(g_t) / (1 + jnp.exp(g_t))
        return f_prime
        
    elif order == 2:
        # Second derivative: f''(t) = exp(g(t))*g'(t)/(1 + exp(g(t)))^2
        e_gt = jnp.exp(g_t)
        denominator = (1 + e_gt) ** 2
        
        # Compute g'(t)
        delta_g_user_prime = calculate_delta_g(
            t=config._t, p=config.p_user, 
            M=config.M_user, T=config.T, 
            n=config.n_user, order=1,
            pulse_time=config.pulse_time,
            pulse_amp=config.pulse_amp,
            pulse_width=config.pulse_width
        )
        
        delta_g_opt_prime = calculate_delta_g(
            t=config._t, p=p_opt,
            M=config.M_opt, T=config.T,
            n=config.n_opt, order=1
        )
        
        g_prime = delta_g_user_prime + delta_g_opt_prime
        f_double_prime = (e_gt * g_prime) / denominator
        return f_double_prime
        
    elif order == 3:
        # Third derivative involves chain rule and product rule
        e_gt = jnp.exp(g_t)
        denominator = (1 + e_gt) ** 3
        
        # Get g'(t), g''(t), and g'''(t)
        derivatives = []
        for derivative_order in range(1, 4):
            delta_g_user_n = calculate_delta_g(
                t=config._t, p=config.p_user,
                M=config.M_user, T=config.T,
                n=config.n_user, order=derivative_order,
                pulse_time=config.pulse_time,
                pulse_amp=config.pulse_amp,
                pulse_width=config.pulse_width
            )
            
            delta_g_opt_n = calculate_delta_g(
                t=config._t, p=p_opt,
                M=config.M_opt, T=config.T,
                n=config.n_opt, order=derivative_order
            )
            
            derivatives.append(delta_g_user_n + delta_g_opt_n)
        
        g_prime, g_double_prime, g_triple_prime = derivatives
        
        # Computing f'''(t) using the chain rule
        term1 = e_gt * g_triple_prime
        term2 = e_gt * g_prime * g_double_prime
        term3 = -2 * e_gt * g_prime ** 3
        
        f_triple_prime = (term1 + term2 + term3) / denominator
        return f_triple_prime
        
    else:
        raise ValueError("Order higher than 3 is not implemented.")

################################################################################
# Schwarzian Action Computation
################################################################################

def schwarzian_derivative(p_opt, config: PerturbationConfig):
    """
    Compute Schwarzian derivative S(f, t).

    Parameters
    ----------
    p_opt : jnp.ndarray
        Optimizer-controlled Fourier coefficients
    config : PerturbationConfig
        User and optimizer parameters

    Returns
    -------
    jnp.ndarray
        Schwarzian derivative S(f) = f'''/f' - (3/2)(f''/f')²
    """
    fp = calculate_f(p_opt, config, order=1)
    fpp = calculate_f(p_opt, config, order=2)
    fppp = calculate_f(p_opt, config, order=3)
    S = fppp / fp - 1.5 * (fpp / fp) ** 2
    return S

def schwarzian_action(p_opt, config: PerturbationConfig):
    """
    Compute Schwarzian action via integration.

    Parameters
    ----------
    p_opt : jnp.ndarray
        Optimizer-controlled Fourier coefficients
    config : PerturbationConfig
        User and optimizer parameters

    Returns
    -------
    float
        Action value -C∫S(f)dt
    """
    S = schwarzian_derivative(p_opt, config)
    action = -config.C * jax.scipy.integrate.trapezoid(S, config._t)
    return action

def action_to_minimize(p_opt, config: PerturbationConfig):
    """
    Objective function computing Schwarzian action.

    Parameters
    ----------
    p_opt : jnp.ndarray
        Optimizer-controlled Fourier coefficients
    config : PerturbationConfig
        System configuration parameters

    Returns
    -------
    float
        Schwarzian action value

    See Also
    --------
    schwarzian_action
    """
    return schwarzian_action(p_opt, config)

################################################################################
# Optimization Functionality
################################################################################

def run_optimizations(action_to_minimize, p_initial, config, verbose=False):
    """
    Execute optimization methods based on configuration.

    Parameters
    ----------
    action_to_minimize : callable
        Target function for optimization
    p_initial : jnp.ndarray
        Initial parameter guess
    config : dict
        Optimizer configuration flags
    verbose : bool, default=False
        Whether to print optimization progress

    Returns
    -------
    dict
        Results for each optimizer including:
        - optimized_params
        - action_values
        - times_taken
    """
    optimization_methods = [
        ("BFGS", run_bfgs_optimization, (action_to_minimize, p_initial, verbose)) if config["BFGS"] else None,
        ("Adam (JAX)", run_adam_optimization, (action_to_minimize, p_initial, verbose)) if config["Adam (JAX)"] else None,
        ("Adam (Optax)", run_optax_adam_optimization, (action_to_minimize, p_initial, verbose)) if config["Adam (Optax)"] else None,
        ("Yogi", run_yogi_optimization, (action_to_minimize, p_initial, verbose)) if config["Yogi"] else None,
        ("LBFGS", run_lbfgs_optimization, (action_to_minimize, p_initial, verbose)) if config["LBFGS"] else None,
        ("AdaBelief", run_adabelief_optimization, (action_to_minimize, p_initial, verbose)) if config["AdaBelief"] else None,
        ("Newton's Method", run_newtons_method, (action_to_minimize, p_initial, verbose)) if config["Newton's Method"] else None,
        ("Hessian-based Optimization", run_hessian_optimization, (action_to_minimize, p_initial, verbose)) if config["Hessian-based Optimization"] else None
    ]

    optimization_methods = [method for method in optimization_methods if method is not None]

    results = {
        "optimized_params": {},
        "action_values": {},
        "times_taken": {}
    }

    for method_name, optimization_function, args in optimization_methods:
        if verbose:
            print(f"\n{'-'*60}")
            print(f"Running {method_name}")
            print(f"{'-'*60}")
        
        p_optimal, action_value, time_taken = optimization_function(*args)
        
        # Skip storing NaN results
        if jnp.isnan(action_value):
            if verbose:
                print(f"{method_name} failed to converge.")
            continue
            
        results["optimized_params"][method_name] = p_optimal
        results["action_values"][method_name] = action_value
        results["times_taken"][method_name] = time_taken
        
        if verbose:
            print(f"{method_name} completed in {time_taken:.4f}s (Action: {action_value:.6f})")

    return results

def reparameterise_ft(action_to_minimize, p_initial, config, pert_config, verbose=False):
    """
    Optimize f(t) reparameterization with full results.

    Parameters
    ----------
    action_to_minimize : callable
        Target function for optimization
    p_initial : jnp.ndarray
        Initial parameter guess
    config : dict
        Optimizer configuration flags
    pert_config : PerturbationConfig
        Parameters for f(t) calculation
    verbose : bool, default=False
        Whether to print optimization progress

    Returns
    -------
    dict
        Complete optimization results including:
        - optimized_params
        - action_value
        - time_taken
        - f_t values
    """
    if verbose:
        print("\nStarting reparametrization optimization...")
    
    # Run optimizations and get base results with verbose parameter
    results = run_optimizations(action_to_minimize, p_initial, config, verbose=verbose)

    # Calculate f(t) for each optimizer's result and add to results
    results["f_t"] = {}
    for method_name, p_optimal in results["optimized_params"].items():
        if verbose:
            print(f"\nCalculating f(t) for {method_name}...")
        f_t_values = calculate_f(p_optimal, pert_config, order=0)
        results["f_t"][method_name] = f_t_values
        if verbose:
            print(f"f(t) calculation completed for {method_name}")

    if verbose:
        print("\nReparametrization optimization completed.")

    return results
