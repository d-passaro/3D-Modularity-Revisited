"""
Title: Weil Representation Functions (Improved Version)

This module contains functions for computing Weil representations and related
modular forms for 3D modularity. The functions implement mathematical
objects used in the study of mock theta functions and their modular properties.

Key components:
- Weil representation matrices and projectors
- False theta series and mock theta functions
- Indefinite theta functions
- Ramanujan's order 7 mock theta functions

Mathematical background: These functions implement results from algebraic
number theory and modular forms, particularly focused on Weil representations
of quadratic forms and their applications to mock theta functions.
"""

from typing import List, Tuple, Union, Optional, Any
import numpy as np
import pandas as pd
import itertools
from sage.combinat.q_analogues import q_pochhammer

# Type aliases for clarity
Matrix = Any  # SageMath matrix type
Vector = Any  # SageMath vector type
PowerSeries = Any  # SageMath power series type
SymbolicExpression = Any  # SageMath symbolic expression type

# Global data: Load precomputed P polynomial data
p_polys = pd.read_pickle("P_poly.pkl")


def omega(m: int, n: int) -> Matrix:
    """
    Compute the matrix omega_{m,n} of the Weil representation.

    This function constructs a fundamental matrix in the Weil representation
    that encodes the action of certain transformations on theta functions.

    Args:
        m (int): Primary modulus parameter
        n (int): Secondary modulus parameter, must divide m

    Returns:
        Matrix: A (2m × 2m) matrix representing omega_{m,n}

    Mathematical Reference:
        See [1] around equation (2.35) for mathematical details.

    Notes:
        The matrix entries are determined by congruence conditions:
        omega[i,j] = 1 if (i+j) ≡ 0 (mod 2n) and (i-j) ≡ 0 (mod 2m/n)
        omega[i,j] = 0 otherwise
    """
    def omega_entry(i: int, j: int) -> int:
        """Compute individual matrix entry based on congruence conditions."""
        condition1 = ((i + j) % (2 * n) == 0)
        condition2 = ((i - j) % (2 * m // n) == 0)
        return int(condition1 and condition2)

    return matrix(2 * m, 2 * m, omega_entry)


def p_plus(m: int, n: int) -> Matrix:
    """
    Compute the positive projection matrix P_{m,n}^+ of the Weil representation.

    This projector extracts the '+1 eigenspace' of the omega matrix, which
    corresponds to certain symmetries in the theta functions.

    Args:
        m (int): Primary modulus parameter
        n (int): Secondary modulus parameter

    Returns:
        Matrix: The positive projection matrix P_{m,n}^+ = (I + omega_{m,n})/2

    Mathematical Reference:
        See [1] around equation (2.36) for details.
    """
    return (identity_matrix(2 * m) + omega(m, n)) / 2


def p_minus(m: int, n: int) -> Matrix:
    """
    Compute the negative projection matrix P_{m,n}^- of the Weil representation.

    This projector extracts the '-1 eigenspace' of the omega matrix, which
    corresponds to antisymmetric components in the theta functions.

    Args:
        m (int): Primary modulus parameter
        n (int): Secondary modulus parameter

    Returns:
        Matrix: The negative projection matrix P_{m,n}^- = (I - omega_{m,n})/2

    Mathematical Reference:
        See [1] around equation (2.36) for details.
    """
    return (identity_matrix(2 * m) - omega(m, n)) / 2


def weil_projector(m: int, K: List[int], irrep: bool = True) -> np.ndarray:
    """
    Compute the complete projector for the Weil representation for subset K.

    This function constructs the main projector by composing positive projections
    for each element in K, followed by optional irreducible representation
    adjustments and a final negative projection.

    Args:
        m (int): Primary modulus parameter
        K (List[int]): Subset of divisors to project onto
        irrep (bool): Whether to apply irreducible representation corrections

    Returns:
        np.ndarray: The complete Weil projector matrix

    Mathematical Reference:
        See [1] around equation (2.39) for details.

    Algorithm:
        1. Start with identity matrix
        2. Apply positive projections for each n in K
        3. If irrep=True, apply corrections for square divisors
        4. Apply final negative projection
    """
    # Initialize with identity matrix
    projector = np.eye(2 * m)

    # Apply positive projections for each element in K
    for n in K:
        projector = projector @ p_plus(m, n)

    # Apply irreducible representation corrections if requested
    if irrep:
        for f in divisors(m):
            # Check if f^2 divides m and f ≠ 1
            if f**2 in divisors(m) and f != 1:
                correction = np.eye(2 * m) - omega(m, f) / f
                projector = projector @ correction

    # Apply final negative projection
    projector = projector @ p_minus(m, m)

    return projector


def weil_reps(m: int, K: List[int], irrep: bool = True) -> Tuple[List[List[int]], List[List[int]]]:
    """
    Extract Weil representation data from the projector.

    This function computes the projector and extracts the non-zero components
    along with their signs, identifying the irreducible representations.

    Args:
        m (int): Primary modulus parameter
        K (List[int]): Subset of divisors
        irrep (bool): Whether to use irreducible representation

    Returns:
        Tuple[List[List[int]], List[List[int]]]:
            - List of index lists for each representation
            - List of sign lists for each representation

    Notes:
        This function identifies unique representations by finding rows of the
        projector with non-zero entries and extracting their support and signs.
    """
    projector = weil_projector(m, K, irrep=irrep)

    # Extract indices and signs for non-zero entries in each row
    indices_list = []
    signs_list = []

    for row in projector:
        non_zero_mask = (row != 0)
        if np.any(non_zero_mask):
            indices = np.arange(2 * m)[non_zero_mask]
            signs = np.sign(row[non_zero_mask])
            indices_list.append(indices)
            signs_list.append(signs)

    # Remove duplicate representations
    unique_reps = []
    unique_signs = []

    for rep_indices, rep_signs in zip(indices_list, signs_list):
        rep_list = list(rep_indices)
        if rep_list not in unique_reps:
            unique_reps.append(rep_list)
            unique_signs.append(list(rep_signs))

    return unique_reps, unique_signs


def false_theta(m: int, r: int, max_n: int, q: SymbolicExpression = var("q")) -> SymbolicExpression:
    """
    Compute the false theta series ψ_{m,r}(q).

    False theta series are q-series that appear in the study of mock theta
    functions and have applications in partition theory and modular forms.

    Args:
        m (int): Modulus parameter
        r (int): Residue class parameter
        max_n (int): Maximum summation index (series truncated at ±max_n)
        q (SymbolicExpression): Formal variable (default: var("q"))

    Returns:
        SymbolicExpression: The false theta series
        ψ_{m,r}(q) = Σ_{n=-max_n}^{max_n} sign(r + 2mn) * q^((r + 2mn)²/(4m))

    Notes:
        The 'round(sign(k))' gives 0 for k=0, ±1 for k≠0, which differs from
        the standard sign function that gives 0 for k=0.
    """
    psi = 0
    for n in range(-max_n, max_n + 1):
        k = r + 2 * m * n
        coefficient = round(sign(k))  # Rounded sign to handle k=0 case
        psi += coefficient * q**(k**2 / (4 * m))

    return psi


def Psi(m: int, r: int, max_n: int, q: SymbolicExpression = var("q")) -> SymbolicExpression:
    """
    Compute the Psi theta series (alternative form).

    This is similar to false_theta but uses the standard sign function
    instead of the rounded version.

    Args:
        m (int): Modulus parameter
        r (int): Residue class parameter
        max_n (int): Maximum summation index
        q (SymbolicExpression): Formal variable

    Returns:
        SymbolicExpression: The Psi series
        Ψ_{m,r}(q) = Σ_{n=-max_n}^{max_n} sign(r + 2mn) * q^((r + 2mn)²/(4m))
    """
    psi = 0
    for n in range(-max_n, max_n + 1):
        k = r + 2 * m * n
        psi += sign(k) * q**(k**2 / (4 * m))

    return psi


def unary_theta(m: int, r: int, max_n: int, q: SymbolicExpression = var("q")) -> SymbolicExpression:
    """
    Compute the unary theta series (weighted by coefficients).

    This is a variant of the theta series where terms are weighted by their
    coefficient values rather than just signs.

    Args:
        m (int): Modulus parameter
        r (int): Residue class parameter
        max_n (int): Maximum summation index
        q (SymbolicExpression): Formal variable

    Returns:
        SymbolicExpression: The unary theta series
        θ_{m,r}(q) = Σ_{n=-max_n}^{max_n} (r + 2mn) * q^((r + 2mn)²/(4m))
    """
    theta = 0
    for n in range(-max_n, max_n + 1):
        k = r + 2 * m * n
        theta += k * q**(k**2 / (4 * m))

    return theta


def format_expansion(expansion: PowerSeries, order: int = 20) -> PowerSeries:
    """
    Format a power series expansion for improved readability.

    This function normalizes a power series by factoring out the leading term
    and presenting it in a more readable form.

    Args:
        expansion (PowerSeries): The power series to format
        order (int): Order for series expansion (default: 20)

    Returns:
        PowerSeries: Formatted expansion as leading_coeff * q^leading_power * (normalized_series)

    Notes:
        The function extracts the first coefficient and power, then factors them out
        to present the series in normalized form.
    """
    # Extract first coefficient and power
    first_coeff, first_power = expansion.coefficients(q)[0]
    first_coeff = int(first_coeff)

    # Normalize and format
    normalized_series = expand(1/first_coeff * q**(-first_power) * expansion)
    formatted = int(first_coeff) * q**(first_power) * normalized_series.series(q, order)

    return formatted


def P_poly(n: int, p: int, b: int) -> Union[int, SymbolicExpression]:
    """
    Load precomputed P polynomial for given parameters.

    P polynomials are special polynomials that appear in the study of
    mock theta functions and their modular properties.

    Args:
        n (int): Primary parameter
        p (int): Secondary parameter
        b (int): Tertiary parameter

    Returns:
        Union[int, SymbolicExpression]: The P polynomial value

    Raises:
        ValueError: If polynomial with specified parameters is not found

    Notes:
        Returns 1 for trivial cases (n < 2 or p = 0).
        For other cases, looks up precomputed values from the pandas DataFrame.
    """
    # Handle trivial cases
    if n < 2:
        return 1
    if p == 0:
        return 1

    # Look up in precomputed data
    try:
        condition = (p_polys["n"] == n) & (p_polys["p"] == p) & (p_polys["b"] == b)
        result = p_polys.loc[condition, "P_poly"].values[0]
        return result
    except (IndexError, KeyError):
        raise ValueError(f'P polynomial with parameters n={n}, p={p}, b={b} is missing from data')


def indefinite_theta(A: Matrix, a: Vector, b: Vector, c1: Vector, c2: Vector, n_max: int) -> SymbolicExpression:
    """
    Compute indefinite theta functions.

    Indefinite theta functions are generalizations of classical theta functions
    to indefinite quadratic forms, important in the theory of mock theta functions.

    Args:
        A (Matrix): Quadratic form matrix (typically 2×2)
        a (Vector): Translation vector
        b (Vector): Phase vector
        c1 (Vector): First sign vector for rho function
        c2 (Vector): Second sign vector for rho function
        n_max (int): Maximum summation range

    Returns:
        SymbolicExpression: The indefinite theta function
        θ(A,a,b,c1,c2) = Σ_n ρ(n+a) * q^((n+a)ᵀA(n+a)/2) * exp(2πi nᵀAb)

    Notes:
        The rho function ρ(n) = sign(c1ᵀAn) - sign(c2ᵀAn) provides the signing
        that makes the series convergent despite the indefinite form.
    """
    def rho(n: Vector) -> int:
        """Compute the rho function for signing."""
        return sign(c1 * A * n) - sign(c2 * A * n)

    theta = 0
    # Sum over all lattice points in the specified range
    for n_tuple in itertools.product(range(-n_max, n_max + 1), repeat=2):
        n = vector(n_tuple)
        shifted_n = n + a

        # Compute the theta function term
        rho_value = rho(shifted_n)
        quadratic_term = q**((shifted_n * A * shifted_n) / 2)
        phase_term = exp(2 * pi * i * n * A * b)

        theta += rho_value * quadratic_term * phase_term

    return theta


def zhat_indefinite_theta(p: List[int], x: int, r: int, chi: int,
                         n_max: int = 10, nu: List[int] = [0, 0, 0],
                         c2: Optional[Vector] = None) -> PowerSeries:
    """
    Compute the Z-hat indefinite theta function.

    This is a specialized indefinite theta function that appears in the study
    of mock theta functions and their modular transformation properties.

    Args:
        p (List[int]): List of three prime parameters
        x (int): Primary modular parameter
        r (int): Ramanujan parameter
        chi (int): Character parameter
        n_max (int): Maximum summation range (default: 10)
        nu (List[int]): Perturbation parameters (default: [0,0,0])
        c2 (Optional[Vector]): Optional second sign vector

    Returns:
        PowerSeries: The normalized Z-hat function as a power series

    Notes:
        This function computes a ratio of indefinite theta functions normalized
        by a Ramanujan theta function, which yields modular objects related to
        mock theta functions.
    """
    # Compute basic parameters
    m = product(p)
    c1 = vector([1, 0])

    # Set default c2 if not provided
    if c2 is None:
        c2 = vector([x, 2 * (6 * r + 1)])

    # Define vectors and matrix for indefinite form
    b = vector([0, 1 / (2 * x)])
    A = matrix([[-2 * m, 0], [0, x]])

    # Compute the list of r values based on perturbations
    r_values = []
    for eps in itertools.product([1, -1], repeat=3):
        r_val = m - sum(eps[i] * (1 + nu[i]) * m / p[i] for i in range(3))
        r_values.append(r_val)

    # Compute corresponding a vectors
    a_vectors = [vector([r_val / (2 * m), -chi / (2 * x)]) for r_val in r_values]

    # Sum over all sign combinations
    total = 0
    for eps, a in zip(itertools.product([1, -1], repeat=3), a_vectors):
        sign_product = product(eps)
        total += sign_product * indefinite_theta(A, a, b, c1, c2, n_max)

    # Normalize by Ramanujan theta function
    ramanujan_th = ramanujan_theta(x, chi, n_max)

    # Extract leading terms for normalization
    total_coeff, total_power = total.coefficients(q)[0]
    ramanujan_coeff, ramanujan_power = ramanujan_th.coefficients(q)[0]

    # Compute the normalized Z-hat function
    power_shift = total_power - ramanujan_power
    numerator = (total * q**(-total_power)).expand()
    denominator = (ramanujan_th * q**(-ramanujan_power)).expand()

    z_hat = q**(power_shift) * (numerator / denominator).series(q, n_max)

    return z_hat


def F0(prec: int) -> PowerSeries:
    """
    Compute Ramanujan's order 7 mock theta function F0(q).

    F0 is one of Ramanujan's third-order mock theta functions, which exhibit
    modular-like transformation properties but are not quite modular forms.

    Args:
        prec (int): Precision (maximum power of q to compute)

    Returns:
        PowerSeries: F0(q) = Σ_{n≥0} q^(n²) / (q^(n+1); q)_n

    Mathematical Definition:
        F0(q) = Σ_{n≥0} q^(n²) / (q^(n+1); q)_n
        where (a; q)_n = ∏_{k=0}^{n-1} (1 - aq^k) is the q-Pochhammer symbol.

    Notes:
        The series is truncated based on the constraint n² < prec to ensure
        all computed terms are within the desired precision.
    """
    series_sum = 0
    max_n = ceil(sqrt(prec - 1))  # Ensure n² < prec

    for n in range(max_n + 1):
        numerator = q**(n**2)
        denominator = q_pochhammer(n, q**(n + 1), q)
        series_sum += numerator / denominator

    return series_sum


def F1(prec: int) -> PowerSeries:
    """
    Compute Ramanujan's order 7 mock theta function F1(q).

    Args:
        prec (int): Precision (maximum power of q to compute)

    Returns:
        PowerSeries: F1(q) = Σ_{n≥0} q^(n²) / (q^n; q)_n

    Mathematical Definition:
        F1(q) = Σ_{n≥0} q^(n²) / (q^n; q)_n
    """
    series_sum = 0
    max_n = ceil(sqrt(prec - 1))

    for n in range(max_n + 1):
        numerator = q**(n**2)
        denominator = q_pochhammer(n, q**n, q)
        series_sum += numerator / denominator

    return series_sum


def F2(prec: int) -> PowerSeries:
    """
    Compute Ramanujan's order 7 mock theta function F2(q).

    Args:
        prec (int): Precision (maximum power of q to compute)

    Returns:
        PowerSeries: F2(q) = Σ_{n≥0} q^(n²+n) / (q^(n+1); q)_{n+1}

    Mathematical Definition:
        F2(q) = Σ_{n≥0} q^(n²+n) / (q^(n+1); q)_{n+1}
    """
    series_sum = 0
    max_n = ceil(sqrt(prec - 1))

    for n in range(max_n + 1):
        numerator = q**(n**2 + n)
        denominator = q_pochhammer(n + 1, q**(n + 1), q)
        series_sum += numerator / denominator

    return series_sum


def ramanujan_theta(x: int, chi: int, n_max: int) -> SymbolicExpression:
    """
    Compute Ramanujan's theta function.

    This is a classical theta function that appears in the normalization
    of various mock theta functions and modular forms.

    Args:
        x (int): Modular parameter
        chi (int): Character parameter
        n_max (int): Maximum summation range

    Returns:
        SymbolicExpression: The Ramanujan theta function
        θ(x,χ) = Σ_{n=-n_max}^{n_max} (-1)^n * q^(x/2 * (n - χ/(2x))²)

    Notes:
        This function is closely related to classical Jacobi theta functions
        and provides the modular framework for understanding mock theta functions.
    """
    theta = 0
    for n in range(-n_max, n_max + 1):
        coefficient = (-1)**n
        argument = x/2 * (n - chi/(2*x))**2
        theta += coefficient * q**(argument)

    return theta
