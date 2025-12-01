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
    I = np.eye(2)

    ZI = np.kron(Z, I)
    IZ = np.kron(I, Z)
    ZZ = np.kron(Z, Z)

    noisy_rho = (
        (1 - p)**2 * rho
        + p*(1 - p) * ZI @ rho @ ZI
        + p*(1 - p) * IZ @ rho @ IZ
        + p**2 * ZZ @ rho @ ZZ
    )
    return noisy_rho

def amplitude_damping_noise(rho, gamma):
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

    Ks = [
        np.kron(K0, K0),
        np.kron(K0, K1),
        np.kron(K1, K0),
        np.kron(K1, K1)
    ]

    noisy_rho = sum(K @ rho @ K.conj().T for K in Ks)
    return noisy_rho

def bit_flip_noise(rho, p):
    """
    Apply bit-flip noise to a density matrix.

    Args:
        rho (np.ndarray): Density matrix of the quantum state.
        p (float): Probability of bit-flip (0 <= p <= 1).

    Returns:
        np.ndarray: Density matrix after applying bit-flip noise.
    """
    X = np.array([[0, 1], [1, 0]])
    I = np.eye(2)

    XI = np.kron(X, I)
    IX = np.kron(I, X)
    XX = np.kron(X, X)

    noisy_rho = (
        (1 - p)**2 * rho
        + p*(1 - p) * XI @ rho @ XI
        + p*(1 - p) * IX @ rho @ IX
        + p**2 * XX @ rho @ XX
    )
    return noisy_rho