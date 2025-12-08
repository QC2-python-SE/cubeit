"""
Module for handling density matrices (DMs) in the cubeit package.
"""

import numpy as np
from cubeit.noise import *
from itertools import product
from cubeit.gates_DMs import swap

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

    n = int(np.sqrt(rho.shape[0])) # Number of qubits
    
    proj_0z = np.array([[1, 0], [0, 0]]) # Create projector for |0><0|
    proj_1z = np.array([[0, 0], [0, 1]]) # Create projector for |1><1|
    proj_0x = 0.5 * np.array([[1, 1], [1, 1]]) # Create projector for |+><+|
    proj_1x = 0.5 * np.array([[1, -1], [-1, 1]]) # Create projector for |-><-|
    proj_0y = 0.5 * np.array([[1, -1j], [1j, 1]]) # Create projector for |i><i|
    proj_1y = 0.5 * np.array([[1, 1j], [-1j, 1]]) # Create projector for |-i><-i|

    if basis == 'Z':
        if n == 1:
            projectors = [proj_0z, proj_1z]
        elif n == 2:
            projectors = [
                np.kron(proj_0z, proj_0z),
                np.kron(proj_0z, proj_1z),
                np.kron(proj_1z, proj_0z),
                np.kron(proj_1z, proj_1z)
            ]
        else:
            raise ValueError("Only 1 or 2 qubits supported")
    elif basis == 'X':
        if n == 1:
            projectors = [proj_0x, proj_1x]
        elif n == 2:
            projectors = [
                np.kron(proj_0x, proj_0x),
                np.kron(proj_0x, proj_1x),
                np.kron(proj_1x, proj_0x),
                np.kron(proj_1x, proj_1x)
            ]
        else:
            raise ValueError("Only 1 or 2 qubits supported")
    elif basis == 'Y':
        if n == 1:
            projectors = [proj_0y, proj_1y]
        elif n == 2:
            projectors = [
                np.kron(proj_0y, proj_0y),
                np.kron(proj_0y, proj_1y),
                np.kron(proj_1y, proj_0y),
                np.kron(proj_1y, proj_1y)
            ]
        else:
            raise ValueError("Only 1 or 2 qubits supported")
    else:
        raise ValueError("Basis must be one of 'X', 'Y', or 'Z'.")

    outcomes = [''.join(bits) for bits in product('01', repeat=n)] # Generate outcome labels based on number of qubits
    probs = {outcome: np.trace(P @ rho).real for outcome, P in zip(outcomes, projectors)}

    return probs

def DM_measurement_shots(rho: np.ndarray, shots: np.ndarray, basis: str = 'Z'):
    """
    Simulate measurement of a density matrix in a given basis with a finite number of shots.

    Args:
        rho (np.ndarray): Density matrix of the quantum state.
        shots (np.ndarray): Array of measurement shots to simulate.
        basis (str): Measurement basis (One of Pauli X,Y,Z).

    Returns:
        list of dicts: Fractions of each outcome for each shot number.
        dict: Ideal measurement probabilities.
    """

    probs = DM_measurement_ideal(rho, basis)  # ideal probabilities as dict: {'00':0.5, '01':0.0, ...}

    # All possible outcomes as strings
    outcomes_list = list(probs.keys())
    probs_list = [np.abs(probs[k]) for k in outcomes_list]

    shots = [int(s) for s in shots]  # ensure integer
    results = []

    for shot in shots:
        # Sample outcomes according to probabilities
        sampled = np.random.choice(outcomes_list, size=shot, p=probs_list)
        counts = {k: np.sum(sampled == k) / shot for k in outcomes_list}  # fraction of each outcome
        results.append(counts)

    return results, probs

def DM_measurement_shots_noise(rho: np.ndarray, shots: np.ndarray, basis: str ='Z', p_flip_on = False):
    """
    Simulate measurement of a density matrix with readout noise for n qubits.

    Args:
        rho (np.ndarray): Density matrix of the quantum state.
        shots (np.ndarray): Number of measurement shots to simulate.
        basis (str): Measurement basis (One of Pauli X,Y,Z).
        p_flip (dict): Dictionary of bit-flip probabilities for each bit:
            e.g., {'01': 0.02, '10': 0.05} for single qubit. For n qubits,
            the same flip probabilities are applied independently to each qubit.

    Returns:
        list of dicts: Noisy measurement fractions for each outcome per shot number.
        dict: Ideal measurement probabilities for each outcome.
    """

    if p_flip_on is False:
        p_flip = {'p01': 0.00, 'p10': 0.00}
    else:
        p_flip = {'p01': 0.01, 'p10': 0.02}

    # Get ideal probabilities as a dictionary of bitstring outcomes
    probs = DM_measurement_ideal(rho, basis)  # e.g., {'00': 0.5, '01':0, '10':0, '11':0.5}

    outcomes_list = list(probs.keys())
    probs_list = [np.abs(probs[k]) for k in outcomes_list]

    shots = [int(s) for s in shots]  # ensure integer
    results = []

    for shot in shots:
        # Sample ideal outcomes according to probabilities
        sampled = np.random.choice(outcomes_list, size=shot, p=probs_list)

        noisy_sampled = []
        for bitstring in sampled:
            noisy_bits = ''
            for bit in bitstring:
                if bit == '0':
                    noisy_bits += '1' if np.random.rand() < p_flip['p01'] else '0'
                else:  # bit == '1'
                    noisy_bits += '0' if np.random.rand() < p_flip['p10'] else '1'
            noisy_sampled.append(noisy_bits)

        # Count fractions of each outcome
        counts = {k: np.sum(np.array(noisy_sampled) == k)/shot for k in outcomes_list}
        results.append(counts)

    return results, probs

class DensityMatrix1Qubit:
    """
    Class representing a density matrix for one qubit and providing methods for measurement.
    """

    def __init__(self, rho: np.ndarray):
        self.rho = rho
        self.history = [] # Store gates applied for reference

    def __repr__(self):
        return f"DensityMatrix1Qubit(\n{self.rho}\n)"

    def __str__(self):
        return self.__repr__()
    
    def apply_gate(self, gate: tuple, target: int):
        """
        Apply a single-qubit gate to the density matrix.

        Args:
            gate (tuple): A tuple containing the 2x2 unitary matrix and its name.
            target (int): The target qubit index (0 or 1).
        """
        mat, name = gate
        self.rho = mat @ self.rho @ mat.conj().T # Update density matrix with gate application
        self.history.append({"gate": name, "target": target}) # Store gate applied for reference

    def apply_sequence(self, gates: list, targets: list):
        """
        Apply a sequence of gates to the density matrix by sequentially applying each gate.

        Args:
            gates (list): List of matrices representing the gates, along with their names.
            targets (list): List of target qubit indices for each gate.
        """

        for gate, target in zip(gates, targets):
            mat, name = gate
            if mat.shape[0] == mat.shape[1] == 2: # Checking it is square and a single-qubit gate
                self.apply_gate(gate, target)
            else:
                raise ValueError("Gate must be a single-qubit (2x2) unitary matrix.")
            
    def apply_sequence_noise(self, gates: list, targets: list, noise_channels: dict):
        """
        Apply a sequence of gates to a density matrix with noise channels after each individual gate.

        Args:
            gates (list): List of gate functions which each return their matrix and name.
            targets (list): List of target qubit indices for each gate.
            noise_channels (dict): List of noise channels to apply after each gate with corresponding probabilities.
        """

        allowed = {'depolarising', 'dephasing', 'amplitude damping', 'bit flip'} # Define the allowed noise channels
        invalid = set(noise_channels) - allowed # Find the invalid keys

        if invalid:
            raise ValueError(f"Invalid noise channels: {invalid}. \n Allowed channels are: {allowed}.")
        for gate, target in zip(gates, targets):
            mat, name = gate
            if mat.shape[0] == mat.shape[1] == 2: # Checking it is a single-qubit gate
                self.apply_gate(gate, target)
            else:
                raise ValueError("Gate must be a single-qubit (2x2) unitary matrix.")
            # Apply noise channel after each gate
            #if 'depolarising' in noise_channels:
            #    print( "Applying depolarising noise")
            self.rho = depolarising_noise(self.rho, p=noise_channels['depolarising'])
            #elif 'dephasing' in noise_channels:
            #print( "Applying dephasing noise")
            self.rho = dephasing_noise(self.rho, p=noise_channels['dephasing'])
            #elif 'amplitude damping' in noise_channels:
            #print( "Applying amplitude damping noise")
            self.rho = amplitude_damping_noise(self.rho, gamma=noise_channels['amplitude damping'])
            #elif 'bit flip' in noise_channels:
            #print( "Applying bit flip noise")
            self.rho = bit_flip_noise(self.rho, p=noise_channels['bit flip'])
            
    def measure_ideal(self, basis: str ='Z'):
        """
        Return ideal measurement probabilities for this density matrix.
        
        Args:
            basis (str): 'X', 'Y', or 'Z'.

        Returns:
            dict: Ideal measurement probabilities.
        """
        return DM_measurement_ideal(self.rho, basis)
    
    def measure_shots(self, shots: list, basis: str ='Z', pdict: dict = {'p01': 0.02, 'p10': 0.05}):
        """
        Method to simulate noisy measurement of the density matrix.

        Args:
            shots (list or array): Number of measurement shots.
            basis (str): 'X', 'Y', or 'Z'.
            pdict (dict): Dictionary containing readout error probabilities:

        Returns:
            np.ndarray: Noisy measurement outcomes for '0' and '1'.
            dict: Ideal measurement probabilities.
        """
        return DM_measurement_shots_noise(
            self.rho,
            shots,
            basis=basis,
            p_flip_on=False
        )

class DensityMatrix2Qubit:
    """
    Class representing a density matrix for two qubits and providing methods for measurement.
    """

    def __init__(self, rho: np.ndarray):
        self.rho = rho
        self.history = [] # Store gates applied for reference

    def __repr__(self):
        return f"DensityMatrix2Qubit(\n{self.rho}\n)"

    def __str__(self):
        return self.__repr__()

    def apply_single_qubit_gate(self, gate: tuple, target: int):
        """
        Apply a single-qubit gate to the density matrix.

        Args:
            gate (tuple): A tuple containing the 2x2 unitary matrix of a gate and its name.
            target (int): The target qubit index (0 or 1).
        """
        mat, name = gate
        
        if target == 0:
            two_q_gate = np.kron(mat, np.eye(2,dtype=complex)) # Expand gate to act on first qubit
        elif target == 1:
            two_q_gate = np.kron(np.eye(2,dtype=complex), mat) # Expand gate to act on second qubit
        else:
            raise ValueError("Target qubit index must be 0 or 1.")
        
        self.history.append({"gate": name, "target": target}) # Store gate applied for reference

        self.rho = two_q_gate @ self.rho @ two_q_gate.conj().T # Update density matrix with gate application

    def apply_sequence(self, gates: list, targets: list):
        """
        Apply a sequence of gates to the density matrix by sequentially applying each gate.

        Args:
            gates (list): List of matrices representing the gates, along with their names.
            targets (list): List of target qubit indices for each gate.
        """

        for gate, target in zip(gates, targets):
            mat, name = gate
            if mat.shape[0] == mat.shape[1] == 2: # Checking it is square and a single-qubit gate
                self.apply_single_qubit_gate(gate, target)
            elif mat.shape[0] == mat.shape[1] == 4: # Gate is already a two-qubit gate e.g. CX, CZ etc. No capability right now to specify which qubit is target/control
                SWAP, _ = swap() # Get SWAP gate matrix
                if target == [0,1]: # For control on qubit 0, target on qubit 1
                    self.rho = mat @ self.rho @ mat.conj().T # Update density matrix with gate application
                elif target == [1,0]: # For control on qubit 1, target on qubit 0
                    mat = SWAP @ mat @ SWAP # Switch gate into a form where control is on qubit 1, target on qubit 0
                    self.rho = mat @ self.rho @ mat.conj().T # Update density matrix with gate application
                self.history.append({"gate": name, "target": target}) # Add the gate to history
            else:
                raise ValueError("Gate must be either a single-qubit (2x2) or two-qubit (4x4) unitary matrix.")

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
    
    def bloch_vector_from_density(rho):
        """
        Convert a single-qubit density matrix to its Bloch vector representation.   
        Args:
            rho: 2x2 density matrix of the qubit
        Returns:
            3-element array representing the Bloch vector (bx, by, bz)
        """
        # Pauli matrices
        X = np.array([[0,1],[1,0]], dtype=complex)
        Y = np.array([[0,-1j],[1j,0]], dtype=complex)
        Z = np.array([[1,0],[0,-1]], dtype=complex)

        # Calculate Bloch vector components by taking traces with Pauli matrices
        bx = np.real(np.trace(rho @ X))
        by = np.real(np.trace(rho @ Y))
        bz = np.real(np.trace(rho @ Z))
        return np.array([bx, by, bz])

    def apply_sequence_noise(self, gates: list, targets: list, noise_channels: dict):
        """
        Apply a sequence of gates to a density matrix with noise channels after each individual gate.

        Args:
            gates (list): List of gate functions which each return their matrix and name.
            targets (list): List of target qubit indices for each gate.
            noise_channels (dict): List of noise channels to apply after each gate with corresponding probabilities.
        """

        allowed = {'depolarising', 'dephasing', 'amplitude damping', 'bit flip'} # Define the allowed noise channels

        invalid = set(noise_channels) - allowed # Find the invalid keys

        if invalid:
            raise ValueError(f"Invalid noise channels: {invalid}. \n Allowed channels are: {allowed}.")

        for gate, target in zip(gates, targets):
            mat, name = gate
            if mat.shape[0] == mat.shape[1] == 2: # Checking it is a single-qubit gate
                self.apply_single_qubit_gate(gate, target)
            elif mat.shape[0] == mat.shape[1] == 4: # Gate is already a two-qubit gate e.g. CX, CZ etc. No capability right now to specify which qubit is target/control
                SWAP, _ = swap() # Get SWAP gate matrix
                if target == [0,1]: # For control on qubit 0, target on qubit 1
                    self.rho = mat @ self.rho @ mat.conj().T # Update density matrix with gate application
                elif target == [1,0]: # For control on qubit 1, target on qubit 0
                    mat = SWAP @ mat @ SWAP # Switch gate into a form where control is on qubit 1, target on qubit 0
                    self.rho = mat @ self.rho @ mat.conj().T # Update density matrix with gate application
                self.history.append({"gate": name, "target": target}) # Add the gate to history
            else:
                raise ValueError("Gate must be either a single-qubit (2x2) or two-qubit (4x4) unitary matrix.")

            # Apply noise channel after each gate
            if 'depolarising' in noise_channels:
                self.rho = depolarising_noise(self.rho, p=noise_channels['depolarising'])
            elif 'dephasing' in noise_channels:
                self.rho = dephasing_noise(self.rho, p=noise_channels['dephasing'])
            elif 'amplitude damping' in noise_channels:
                self.rho = amplitude_damping_noise(self.rho, gamma=noise_channels['amplitude damping'])
            elif 'bit flip' in noise_channels:
                self.rho = bit_flip_noise(self.rho, p=noise_channels['bit flip'])

    def measure_ideal(self, basis: str ='Z'):
        """
        Return ideal measurement probabilities for this density matrix.
        
        Args:
            basis (str): 'X', 'Y', or 'Z'.
        
        Returns:
            dict: Ideal measurement probabilities.
        """
        return DM_measurement_ideal(self.rho, basis)

    
    def measure_shots(self, shots: list, basis: str ='Z', pdict: dict = {'p01': 0.02, 'p10': 0.05}):
        """
        Method to simulate noisy measurement of the density matrix.

        Args:
            shots (list or array): Number of measurement shots.
            basis (str): 'X', 'Y', or 'Z'.
            pdict (dict): Dictionary containing readout error probabilities:

        Returns:
            np.ndarray: Noisy measurement outcomes for '0' and '1'.
            dict: Ideal measurement probabilities.
        """
        return DM_measurement_shots_noise(
            self.rho,
            shots,
            basis=basis,
            p_flip_on=False
        )
    
    def clean(self, tol: float = 1e-12):
        """
        Set any entries with real or imaginary part < tol to 0.

        Args:
            tol (float): The tolerance below which real or imaginary parts are rounded to zero
        """
        real = np.where(np.abs(self.rho.real) < tol, 0, self.rho.real)
        imag = np.where(np.abs(self.rho.imag) < tol, 0, self.rho.imag)
        self.rho = real + 1j*imag