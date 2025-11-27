"""
Module for handling density matrices (DMs) in the cubeit package.
"""

import numpy as np
from cubeit.noise import *

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

def DM_measurement_shots(rho: np.ndarray, shots: np.ndarray, basis: str = 'Z'):
    """
    Simulate measurement of a density matrix in a given basis with a finite number of shots.

    Args:
        rho (np.ndarray): Density matrix of the quantum state.
        shots (np.ndarray): Array of measurement shots to simulate.
        basis (str): Measurement basis (One of Pauli X,Y,Z).

    Returns:
        np.ndarray: Noisy measurement outcomes for '0' and '1'.
        dict: Ideal measurement probabilities.
    """

    probs = DM_measurement_ideal(rho, basis) # Get ideal measurement probabilities
    
    shots = [int(s) for s in shots] # Ensure shots are integers
    noisy_0 = np.empty(len(shots)) # Intialise arrays to hold noisy measurement results
    noisy_1 = np.empty(len(shots))
    
    for idx, shot in enumerate(shots):
        outcomes = np.random.choice(['0', '1'], size=shot, p=[probs['0'], probs['1']]) # Sample from the probability distribution with measurement shots
        counts = {'0': np.sum(outcomes == '0'), '1': np.sum(outcomes == '1')} # Count occurrences of each outcome
        noisy_0[idx] = counts['0'] / shot # Store the fraction of '0' and '1' outcomes
        noisy_1[idx] = counts['1'] / shot

    return noisy_0, noisy_1, probs

def DM_measurement_shots_noisy(rho: np.ndarray, shots: np.ndarray, basis: str ='Z', p01: float =0.02, p10: float =0.05):
    """
    Simulate measurement of a density matrix with readout noise.
    
    Args:
        rho (np.ndarray): Density matrix of the quantum state.
        shots (np.ndarray): Number of measurement shots to simulate.
        basis (str): Measurement basis (One of Pauli X,Y,Z).
        p01 (float): Probability of misreading '0' as '1'.
        p10 (float): Probability of misreading '1' as '0'.

    Returns:
        np.ndarray: Noisy measurement outcomes for '0' and '1'.
        dict: Ideal measurement probabilities.
    """

    probs = DM_measurement_ideal(rho, basis) # Ideal projective measurement probabilities

    shots = [int(s) for s in shots] # Ensure shots are integers
    noisy_0 = np.empty(len(shots)) # Intialise arrays to hold noisy measurement results
    noisy_1 = np.empty(len(shots))

    for idx, shot in enumerate(shots):
        ideal = np.random.choice(['0','1'], size=shot, p=[probs['0'], probs['1']]) # Sample ideal outcomes

        noisy = []
        for outcome in ideal: # Apply readout noise to each non ideal outcome
            if outcome == '0':
                noisy.append('1' if np.random.rand() < p01 else '0')
            else:  # outcome == '1'
                noisy.append('0' if np.random.rand() < p10 else '1')
        
        counts = {'0': np.sum(noisy == '0'), '1': np.sum(noisy == '1')} # Count occurrences of each outcome

        noisy_0[idx] = counts['0'] / shot # Store the fraction of '0' and '1' outcomes for this number of shots
        noisy_1[idx] = counts['1'] / shot

    # Count outcomes
    return noisy_0, noisy_1, probs

class DensityMatrix2Qubit:
    """
    Class representing a density matrix and providing methods for measurement.
    """

    def __init__(self, rho: np.ndarray):
        self.rho = rho

    def __repr__(self):
        return f"DensityMatrix2Qubit(\n{self.rho}\n)"

    def __str__(self):
        return self.__repr__()

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
        Apply a sequence of single-qubit gates to the density matrix by sequentially applying each gate.

        Args:
            gates (list): List of 2x2 unitary matrices representing the gates.
            targets (list): List of target qubit indices for each gate.
        """

        for gate, target in zip(gates, targets):
            if gate.shape[0] == 2: # Checking it is a single-qubit gate
                self.apply_single_qubit_gate(gate, target)
            elif gate.shape[0] == 4: # Gate is already a two-qubit gate e.g. CX, CZ etc.
                self.rho = gate @ self.rho @ gate.conj().T # Update density matrix with gate application
            else:
                raise ValueError("Gate must be either a single-qubit (2x2) or two-qubit (4x4) unitary matrix.")

    def apply_sequence2(self, gates: list, targets: list):
        """
        Apply a sequence of gates to a density matrix by multiplying the gates together first.

        Args:
            gates (np.ndarray): Array of 2x2 unitary matrices representing the gates.
            targets (list): List of target qubit indices for each gate.
        """
        
        dim = self.rho.shape[0]

        total_gate = np.eye(dim, dtype=complex) # Initialize total gate as identity for 2 qubits

        for gate, target in zip(gates, targets):
            if gate.shape[0] == 2: # Checking it is a single-qubit gate
                if target == 0:
                    two_q_gate = np.kron(gate, np.eye(2,dtype=complex)) # Expand gate to act on first qubit
                elif target == 1:
                    two_q_gate = np.kron(np.eye(2,dtype=complex), gate) # Expand gate to act on second qubit
                else:
                    raise ValueError("Target qubit index must be 0 or 1.")
            elif gate.shape[0] == 4: # Gate is already a two-qubit gate e.g. CX, CZ etc.
                two_q_gate = gate
            else:
                raise ValueError("Gate must be either a single-qubit (2x2) or two-qubit (4x4) unitary matrix.")

            total_gate = two_q_gate @ total_gate # Multiply gates together

        self.rho = total_gate @ self.rho @ total_gate.conj().T # Update density matrix with total gate application

    def partial_trace(self, keep: list):
        """
        Perform partial trace on a density matrix.
        Args:
            keep: list of indices to keep, e.g. [0, 2] to keep subsystems 0 and 2
        
        Returns:
            reduced density matrix after tracing out unwanted subsystems
        """
        dims = [2] * self.rho.shape[0]
        N = len(dims)
        reshaped = self.rho.reshape(dims + dims) # for a two qubit system this will reshape from (4,4) to (2,2,2,2)
        traced = reshaped
        for i in reversed(range(N)): # looping backwards avoids messing up the axis indices
            if i not in keep:
                traced = np.trace(traced, axis1=i, axis2=i+N)
        return traced
    

    def apply_sequence_noise(self, gates: list, targets: list, noise_channels: dict):
        """
        Apply a sequence of gates to a density matrix with noise channels after each individual gate.

        Args:
            gates (list): List of 2x2 unitary matrices representing the gates.
            targets (list): List of target qubit indices for each gate.
            noise_channels (dict): List of noise channels to apply after each gate with corresponding probabilities.
        """

        allowed = {'depolarising', 'dephasing', 'amplitude_damping', 'bit flip'} # Define the allowed noise channels

        invalid = set(noise_channels) - allowed # Find the invalid keys

        if invalid:
            raise ValueError(f"Invalid noise channels: {invalid}. \n Allowed channels are: {allowed}.")

        for gate, target in zip(gates, targets):
            if gate.shape[0] == 2: # Checking it is a single-qubit gate
                self.apply_single_qubit_gate(gate, target)
            elif gate.shape[0] == 4: # Gate is already a two-qubit gate e.g. CX, CZ etc.
                self.rho = gate @ self.rho @ gate.conj().T # Update density matrix with gate application
            else:
                raise ValueError("Gate must be either a single-qubit (2x2) or two-qubit (4x4) unitary matrix.")

            # Apply noise channel after each gate
            if 'depolarising' in noise_channels:
                self.rho = depolarising_noise(self.rho, p=noise_channels['depolarising'])
            elif 'dephasing' in noise_channels:
                self.rho = dephasing_noise(self.rho, p=noise_channels['dephasing'])
            elif 'amplitude_damping' in noise_channels:
                self.rho = amplitude_damping_noise(self.rho, gamma=noise_channels['amplitude_damping'])
            elif 'bit flip' in noise_channels:
                self.rho = bit_flip_noise(self.rho, p=noise_channels)

    def measure_ideal(self, basis='Z'):
        """
        Return ideal measurement probabilities for this density matrix.
        
        Args:
            basis (str): 'X', 'Y', or 'Z'.
        
        Returns:
            dict: Ideal measurement probabilities.
        """
        return DM_measurement_ideal(self.rho, basis)

    
    def DM_measurement_shots_noisy(self, shots, basis='Z', p01=0.02, p10=0.05):
        """
        Method to simulate noisy measurement of the density matrix.

        Args:
            shots (list or array): Number of measurement shots.
            basis (str): 'X', 'Y', or 'Z'.
            p01 (float): Probability of misreading '0' as '1'.
            p10 (float): Probability of misreading '1' as '0'.

        Returns:
            np.ndarray: Noisy measurement outcomes for '0' and '1'.
            dict: Ideal measurement probabilities.
        """
        return DM_measurement_shots_noisy(
            self.rho,
            shots,
            basis=basis,
            p01=p01,
            p10=p10
        )
    
    def clean(self, tol=1e-10):
        """
        Set any entries with real or imaginary part < tol to 0.
        """
        real = np.where(np.abs(self.rho.real) < tol, 0, self.rho.real)
        imag = np.where(np.abs(self.rho.imag) < tol, 0, self.rho.imag)
        self.rho = real + 1j*imag
