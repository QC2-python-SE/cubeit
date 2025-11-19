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
    density_matrix = state_vector @ state_vector.conj().T # Calculate outer product |psi><psi|
    return density_matrix

def DM_measurement_ideal(rho: np.ndarray, basis: str ='Z'):
    """
    Measure a density matrix in a given basis.

    Args:
        rho (np.ndarray): Density matrix of the quantum state.
        basis (str): Measurement basis (One of Pauli X,Y,Z).
    
    Returns:
        dict: Measurement outcomes and their probabilities.
    """

    if basis == 'Z':
        proj_0 = np.array([[1, 0], [0, 0]]) # Create projector for |0><0|
        proj_1 = np.array([[0, 0], [0, 1]]) # Create projector for |1><1|
    elif basis == 'X':
        proj_0 = 0.5 * np.array([[1, 1], [1, 1]]) # Create projector for |+><+|
        proj_1 = 0.5 * np.array([[1, -1], [-1, 1]]) # Create projector for |-><-|
    elif basis == 'Y':
        proj_0 = 0.5 * np.array([[1, -1j], [1j, 1]]) # Create projector for |i><i|
        proj_1 = 0.5 * np.array([[1, 1j], [-1j, 1]]) # Create projector for |-i><-i|
    else:
        raise ValueError("Basis must be one of 'X', 'Y', or 'Z'.")

    p_0 = np.trace(proj_0 @ rho).real # Calculate probability for outcome 0
    p_1 = np.trace(proj_1 @ rho).real # Calculate probability for outcome 1

    return {'0': p_0, '1': p_1}

def DM_measurement_noise(rho: np.ndarray, shots: int = 1024, basis: str = 'Z'):
    """
    Simulate noisy measurement of a density matrix in a given basis.

    Args:
        rho (np.ndarray): Density matrix of the quantum state.
        shots (int): Number of measurement shots to simulate.
        basis (str): Measurement basis (One of Pauli X,Y,Z).

    Returns:
        dict: Measurement outcomes and their counts.
    """
    probabilities = DM_measurement_ideal(rho, basis)
    outcomes = np.random.choice(['0', '1'], size=shots, p=[probabilities['0'], probabilities['1']]) # Sample from the probability distribution with measurement shots
    
    counts = {'0': np.sum(outcomes == '0'), '1': np.sum(outcomes == '1')}
    return counts