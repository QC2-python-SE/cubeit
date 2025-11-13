"""

Python file containing different noise generation functions.

"""

import numpy as np

def depolarising_noise(rho, p):
    """
    Apply depolarising noise to a density matrix.

    Args:
        rho (np.ndarray): Density matrix of the quantum state.
        p (float): Probability of depolarisation (0 <= p <= 1).

    Returns:
        np.ndarray: Density matrix after applying depolarising noise.
    """

    d = rho.shape[0]
    identity = np.eye(d)
    noisy_rho = (1 - p) * rho + p * identity / d
    return noisy_rho

def dephasing_noise(rho, p):
    """
    Apply dephasing noise to a density matrix.

    Args:
        rho (np.ndarray): Density matrix of the quantum state.
        p (float): Probability of dephasing (0 <= p <= 1).

    Returns:
        np.ndarray: Density matrix after applying dephasing noise.
    """

    Z = np.array([[1, 0], [0, -1]])
    d = rho.shape[0]
    noisy_rho = (1 - p) * rho + p * Z @ rho @ Z
    return noisy_rho
