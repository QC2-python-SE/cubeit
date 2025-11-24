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

def DM_measurement_shots(rho: np.ndarray, shots: int = 1024, basis: str = 'Z'):
    """
    Simulate noisy measurement of a density matrix in a given basis.

    Args:
        rho (np.ndarray): Density matrix of the quantum state.
        shots (int): Number of measurement shots to simulate.
        basis (str): Measurement basis (One of Pauli X,Y,Z).

    Returns:
        dict: Measurement outcomes and their counts.
    """
    probabilities = DM_measurement_ideal(rho, basis) # Get ideal measurement probabilities
    outcomes = np.random.choice(['0', '1'], size=shots, p=[probabilities['0'], probabilities['1']]) # Sample from the probability distribution with measurement shots
    
    counts = {'0': np.sum(outcomes == '0'), '1': np.sum(outcomes == '1')} # Count occurrences of each outcome
    return counts

class DensityMatrix:
    """
    Class representing a density matrix and providing methods for measurement.
    """

    def __init__(self, rho: np.ndarray):
        """
        Initialize the DensityMatrix with a state vector.

        Args:
            state_vector (np.ndarray): State vector of the quantum state.
        """
        self.rho = rho

    def apply_single_qubit_gate(self, gate: np.ndarray, target: int):
        """
        Apply a single-qubit gate to the density matrix.

        Args:
            gate (np.ndarray): 2x2 unitary matrix representing the gate.
        """
        # First try this way that sequentially applies gates to the density matrix.

        if target == 0:
            two_q_gate = np.kron(gate, np.eye(2,dtype=complex)) # Expand gate to act on first qubit
        elif target == 1:
            two_q_gate = np.kron(np.eye(2,dtype=complex), gate) # Expand gate to act on second qubit
        else:
            raise ValueError("Target qubit index must be 0 or 1.")

        self.rho = two_q_gate @ self.rho @ two_q_gate.conj().T # Update density matrix with gate application

    def apply_sequence(self, gates: list, targets: list):
        """
        Apply a sequence of single-qubit gates to the density matrix.

        Args:
            gates (list): List of 2x2 unitary matrices representing the gates.
            targets (list): List of target qubit indices for each gate.
        """

        for gate, target in zip(gates, targets):
            self.apply_single_qubit_gate(gate, target)

    def apply_sequence2(self, gates: np.ndarray, targets: list):
        """
        Apply a sequence of gates to a density matrix by multiplying the gates together first.

        Args:
            gates (np.ndarray): Array of 2x2 unitary matrices representing the gates.
            targets (list): List of target qubit indices for each gate.
        """
        
        dim = self.rho.shape[0]
        
        total_gate = np.eye(dim, dtype=complex) # Initialize total gate as identity for 2 qubits

        for gate, target in zip(gates, targets):
            if target == 0:
                two_q_gate = np.kron(gate, np.eye(2,dtype=complex)) # Expand gate to act on first qubit
            elif target == 1:
                two_q_gate = np.kron(np.eye(2,dtype=complex), gate) # Expand gate to act on second qubit
            else:
                raise ValueError("Target qubit index must be 0 or 1.")

            total_gate = two_q_gate @ total_gate # Multiply gates together

        self.rho = total_gate @ self.rho @ total_gate.conj().T # Update density matrix with total gate application