# schwarzian.py

from sympy import diff, simplify, lambdify
from scipy.integrate import quad

def schwarzian_der(f, x):
    """
    Compute the Schwarzian derivative of a function f with respect to x.

    Parameters:
    f : sympy expression
        The function f(x) as a SymPy expression.
    x : sympy symbol
        The independent variable.

    Returns:
    sympy expression
        The Schwarzian derivative S(f)(x).
    """
    # Compute the first three derivatives of f
    f_prime = diff(f, x)
    f_double_prime = diff(f_prime, x)
    f_triple_prime = diff(f_double_prime, x)
    
    # Compute the Schwarzian derivative
    S = f_triple_prime / f_prime - (3/2) * (f_double_prime / f_prime)**2
    
    # Simplify the expression
    S_simplified = simplify(S)
    
    return S_simplified

def schwarzian_action(f, x, t0, t1, C=1, numerical=False, subs=None):
    """
    Compute the Schwarzian action of a function f over the interval [t0, t1].
    
    Parameters:
    f : sympy expression
        The function f(x) as a SymPy expression.
    x : sympy symbol
        The independent variable.
    t0 : float
        The lower limit of integration.
    t1 : float
        The upper limit of integration.
    C : float, optional
        The constant coefficient in the Schwarzian action. Default is 1.
    numerical : bool, optional
        If True, perform numerical integration. If False, attempt symbolic integration.
    subs : dict, optional
        A dictionary of substitutions to be made in the expression before evaluation.
    
    Returns:
    float or sympy expression
        The value of the Schwarzian action.
    """
    # Compute the Schwarzian derivative
    S = schwarzian_der(f, x)
    
    if subs is not None:
        # Substitute numerical values into S and f
        S = S.subs(subs)
        f = f.subs(subs)
    
    if numerical:
        # Convert S to a numerical function
        S_func = lambdify(x, S, modules=['numpy'])
        
        # Perform numerical integration
        integral, error = quad(S_func, t0, t1)
        
        # Compute the action
        action = C * integral
        
        return action
    else:
        # Perform symbolic integration
        from sympy import integrate
        
        # Integrate S over [t0, t1]
        action = C * integrate(S, (x, t0, t1))
        
        # Simplify the result
        action_simplified = simplify(action)
        
        return action_simplified

