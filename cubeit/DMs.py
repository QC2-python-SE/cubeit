"""
Module for handling density matrices (DMs) in the cubeit package.
"""

import numpy as np

def create_density_matrix(state_vector):
    """
    Create a density matrix from a state vector.

    Args:
        state_vector (np.ndarray): State vector of the quantum state.

    Returns:
        np.ndarray: Density matrix corresponding to the state vector.
    """
    state_vector = state_vector.reshape(-1, 1)  # Ensure it's a column vector
    density_matrix = state_vector @ state_vector.conj().T
    return density_matrix

def measure_density_matrix(rho, basis):
    """
    Measure a density matrix in a given basis.

    Args:
        rho (np.ndarray): Density matrix of the quantum state.
        basis (str): Measurement basis (One of the Pauli X,Y,Z must be specified).
    
    Returns:
        dict: Measurement outcomes and their probabilities.
    """

    if basis == 'Z':
        proj_0 = np.array([[1, 0], [0, 0]])
        proj_1 = np.array([[0, 0], [0, 1]])
    elif basis == 'X':
        proj_0 = 0.5 * np.array([[1, 1], [1, 1]])
        proj_1 = 0.5 * np.array([[1, -1], [-1, 1]])
    elif basis == 'Y':
        proj_0 = 0.5 * np.array([[1, -1j], [1j, 1]])
        proj_1 = 0.5 * np.array([[1, 1j], [-1j, 1]])
    else:
        raise ValueError("Basis must be one of 'X', 'Y', or 'Z'.")

    p_0 = np.trace(proj_0 @ rho).real
    p_1 = np.trace(proj_1 @ rho).real

    return {'0': p_0, '1': p_1}
