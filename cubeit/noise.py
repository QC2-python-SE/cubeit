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

def amplitude_damping_noise(rho, gamma, qubits=1):
    """
    Apply amplitude damping noise to a density matrix.

    Args:
        rho (np.ndarray): Density matrix of the quantum state.
        gamma (float): Damping probability (0 <= gamma <= 1).

    Returns:
        np.ndarray: Density matrix after applying amplitude damping noise.
    """

    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]])
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]])
    
    if qubits == 1:
        noisy_rho = K0 @ rho @ K0.T + K1 @ rho @ K1.T
    else:
        I = np.eye(2,dtype=complex)
        K0_2 = np.kron(K0, I)
        K1_2 = np.kron(K1, I)
        noisy_rho = K0_2 @ rho @ K0_2.T + K1_2 @ rho @ K1_2.T
    
    return noisy_rho
