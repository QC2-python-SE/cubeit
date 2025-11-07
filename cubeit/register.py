"""
Core quantum state representation and operations for n-qubit systems.
"""

import numpy as np
from typing import Tuple, Optional, List
import math


class QuantumState:
    """Represents a quantum state vector for n qubits."""
    
    def __init__(self, state_vector: np.ndarray, num_qubits: Optional[int] = None):
        """
        Initialize a quantum state.
        
        Args:
            state_vector: Normalized state vector as numpy array (must be 2^n for n qubits)
            num_qubits: Number of qubits (auto-detected if None)
        """
        if not isinstance(state_vector, np.ndarray):
            state_vector = np.array(state_vector, dtype=complex)
        
        # Normalize the state vector
        norm = np.linalg.norm(state_vector)
        if norm > 1e-10:
            self.state = state_vector / norm
        else:
            raise ValueError("State vector cannot be zero")
        
        # Determine number of qubits
        dim = len(self.state)
        if not (dim & (dim - 1) == 0) or dim == 0:
            raise ValueError(f"State vector dimension {dim} must be a power of 2")
        
        if num_qubits is None:
            self.num_qubits = int(np.log2(dim))
        else:
            self.num_qubits = num_qubits
            if 2 ** num_qubits != dim:
                raise ValueError(f"State vector dimension {dim} does not match {num_qubits} qubits (expected {2**num_qubits})")
    
    def __repr__(self):
        return f"QuantumState(num_qubits={self.num_qubits}, dim={len(self.state)})"
    
    def __str__(self):
        return self.to_string()
    
    def _int_to_binary_string(self, n: int, width: int) -> str:
        """Convert integer to binary string with leading zeros."""
        return format(n, f'0{width}b')
    
    def to_string(self, precision: int = 3, max_terms: int = 10) -> str:
        """
        Convert state to human-readable string.
        
        Args:
            precision: Number of decimal places
            max_terms: Maximum number of terms to display
        """
        terms = []
        dim = len(self.state)
        
        # Find all non-zero terms
        non_zero_indices = [i for i, amp in enumerate(self.state) if abs(amp) > 1e-10]
        
        # Sort by magnitude (descending)
        non_zero_indices.sort(key=lambda i: abs(self.state[i]), reverse=True)
        
        # Display up to max_terms
        for i in non_zero_indices[:max_terms]:
            amp = self.state[i]
            real_part = amp.real
            imag_part = amp.imag
            
            # Format coefficient
            if abs(imag_part) < 1e-10:
                coeff = f"{real_part:.{precision}f}"
            elif abs(real_part) < 1e-10:
                coeff = f"{imag_part:.{precision}f}j"
            else:
                coeff = f"{real_part:.{precision}f}{imag_part:+.{precision}f}j"
            
            # Format basis state
            binary = self._int_to_binary_string(i, self.num_qubits)
            basis_state = "|" + "".join(binary) + "⟩"
            
            terms.append(f"{coeff}{basis_state}")
        
        if len(non_zero_indices) > max_terms:
            terms.append(f"... ({len(non_zero_indices) - max_terms} more terms)")
        
        return " + ".join(terms) if terms else "0"
    
    def probabilities(self) -> np.ndarray:
        """Get measurement probabilities for each basis state."""
        return np.abs(self.state) ** 2
    
    def measure(self) -> Tuple[int, ...]:
        """
        Measure all qubits and return the outcome.
        
        Returns:
            Tuple of measurement results (0 or 1) for each qubit
        """
        probs = self.probabilities()
        dim = len(self.state)
        outcome = np.random.choice(dim, p=probs)
        
        # Convert outcome index to qubit values (binary representation)
        result = tuple(int(bit) for bit in self._int_to_binary_string(outcome, self.num_qubits))
        return result
    
    def fidelity(self, other: 'QuantumState') -> float:
        """Calculate fidelity with another quantum state."""
        if len(self.state) != len(other.state):
            raise ValueError("States must have the same dimension")
        overlap = np.abs(np.vdot(self.state, other.state)) ** 2
        return float(overlap)


class _QuantumRegister:
    """An n-qubit quantum register with universal gate operations."""
    
    def __init__(self, num_qubits: int = 2, initial_state: Optional[QuantumState] = None):
        """
        Initialize an n-qubit quantum register.
        
        Args:
            num_qubits: Number of qubits in the register
            initial_state: Initial quantum state (default: |00...0⟩)
        """
        if num_qubits < 1:
            raise ValueError("Number of qubits must be at least 1")
        
        self.num_qubits = num_qubits
        self.dim = 2 ** num_qubits
        
        if initial_state is None:
            # Default to |00...0⟩ state
            initial_vector = np.zeros(self.dim, dtype=complex)
            initial_vector[0] = 1.0
            self.state = QuantumState(initial_vector, num_qubits)
        else:
            if len(initial_state.state) != self.dim:
                raise ValueError(f"State vector must have {self.dim} components for {num_qubits} qubits")
            self.state = initial_state
    
    def apply_gate(self, gate_matrix: np.ndarray):
        """
        Apply a gate matrix to the quantum state.
        
        Args:
            gate_matrix: Unitary matrix of size (2^n, 2^n) where n is number of qubits
        """
        if gate_matrix.shape != (self.dim, self.dim):
            raise ValueError(f"Gate matrix must be {self.dim}x{self.dim} for {self.num_qubits}-qubit system")
        
        self.state = QuantumState(gate_matrix @ self.state.state, self.num_qubits)
    
    def apply_single_qubit_gate(self, gate_matrix: np.ndarray, qubit: int):
        """
        Apply a single-qubit gate to a specific qubit.
        
        Args:
            gate_matrix: 2x2 unitary matrix
            qubit: Which qubit to apply gate to (0 to num_qubits-1)
        """
        if gate_matrix.shape != (2, 2):
            raise ValueError("Single-qubit gate matrix must be 2x2")
        
        if qubit < 0 or qubit >= self.num_qubits:
            raise ValueError(f"Qubit index must be between 0 and {self.num_qubits - 1}")
        
        # Build tensor product: I ⊗ ... ⊗ I ⊗ U ⊗ I ⊗ ... ⊗ I
        # where U is at position qubit (0-indexed from left, i.e., qubit 0 is leftmost)
        full_gate = self._build_single_qubit_gate(gate_matrix, qubit)
        self.apply_gate(full_gate)
    
    def _build_single_qubit_gate(self, gate: np.ndarray, qubit: int) -> np.ndarray:
        """
        Build the full gate matrix for applying a single-qubit gate to a specific qubit.
        
        Args:
            gate: 2x2 gate matrix
            qubit: Qubit index (0 is the most significant qubit)
        
        Returns:
            Full 2^n x 2^n gate matrix
        """
        I = np.eye(2, dtype=complex)

        mats: List[np.ndarray] = []
        for i in range(self.num_qubits):
            mats.append(gate if i == qubit else I)

        result = mats[0]
        for m in mats[1:]:
            result = np.kron(result, m)
        return result
    
    def _int_to_binary_string(self, n: int, width: int) -> str:
        """Convert integer to binary string with leading zeros."""
        return format(n, f'0{width}b')
    
    def apply_two_qubit_gate(self, gate_matrix: np.ndarray, qubit1: int, qubit2: int):
        """
        Apply a two-qubit gate to specific qubits.
        
        Args:
            gate_matrix: 4x4 unitary matrix
            qubit1: First qubit index
            qubit2: Second qubit index
        """
        if gate_matrix.shape != (4, 4):
            raise ValueError("Two-qubit gate matrix must be 4x4")
        
        if qubit1 == qubit2:
            raise ValueError("Qubit indices must be different")
        
        if qubit1 < 0 or qubit1 >= self.num_qubits or qubit2 < 0 or qubit2 >= self.num_qubits:
            raise ValueError(f"Qubit indices must be between 0 and {self.num_qubits - 1}")
        
        # Build the full gate matrix
        full_gate = self._build_two_qubit_gate(gate_matrix, qubit1, qubit2)
        self.apply_gate(full_gate)
    
    def _build_two_qubit_gate(self, gate: np.ndarray, qubit1: int, qubit2: int) -> np.ndarray:
        """
        Build the full gate matrix for applying a two-qubit gate.
        
        Args:
            gate: 4x4 gate matrix
            qubit1: First qubit index
            qubit2: Second qubit index
        
        Returns:
            Full 2^n x 2^n gate matrix
        """
        # Preserve original ordering; work with copies of indices
        original_qubit1, original_qubit2 = qubit1, qubit2
        if qubit1 == qubit2:
            raise ValueError("Qubit indices must be different")

        if qubit1 > qubit2:
            qubit1, qubit2 = qubit2, qubit1
            # Swap the control/target positions within the gate matrix
            swap_matrix = np.array(
                [[1, 0, 0, 0],
                 [0, 0, 1, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1]], dtype=complex
            )
            gate = swap_matrix @ gate @ swap_matrix

        dim = self.dim
        result = np.zeros((dim, dim), dtype=complex)

        for col in range(dim):
            bits = [int(b) for b in self._int_to_binary_string(col, self.num_qubits)]
            input_idx = bits[qubit1] * 2 + bits[qubit2]

            for out_subspace in range(4):
                out_bits = bits.copy()
                out_bits[qubit1] = (out_subspace // 2) & 1
                out_bits[qubit2] = out_subspace % 2
                row = int("".join(str(b) for b in out_bits), 2)
                result[row, col] = gate[out_subspace, input_idx]

        # If original order was reversed, we already accounted for gate adjustment
        return result
    
    def measure(self) -> Tuple[int, ...]:
        """
        Measure all qubits.
        
        Returns:
            Tuple of measurement results (0 or 1) for each qubit
        """
        return self.state.measure()
    
    def measure_qubit(self, qubit: int) -> int:
        """
        Measure a single qubit (partial measurement).
        
        Args:
            qubit: Which qubit to measure (0 to num_qubits-1)
            
        Returns:
            Measurement result (0 or 1)
        """
        if qubit < 0 or qubit >= self.num_qubits:
            raise ValueError(f"Qubit index must be between 0 and {self.num_qubits - 1}")
        
        # Calculate marginal probabilities
        probs = self.state.probabilities()
        
        # Sum probabilities for states where this qubit is 0 or 1
        prob0 = 0.0
        prob1 = 0.0
        
        for i, prob in enumerate(probs):
            binary = self._int_to_binary_string(i, self.num_qubits)
            bit_value = int(binary[self.num_qubits - 1 - qubit])
            
            if bit_value == 0:
                prob0 += prob
            else:
                prob1 += prob
        
        # Normalize
        total = prob0 + prob1
        if total > 1e-10:
            prob0 /= total
            prob1 /= total
        else:
            raise RuntimeError("Measurement probability is zero")
        
        # Sample and collapse state
        result = np.random.choice([0, 1], p=[prob0, prob1])
        
        # Collapse the state
        new_state = self.state.state.copy()
        for i in range(self.dim):
            binary = self._int_to_binary_string(i, self.num_qubits)
            bit_value = int(binary[self.num_qubits - 1 - qubit])
            
            if bit_value != result:
                new_state[i] = 0
        
        # Normalize
        norm = np.linalg.norm(new_state)
        if norm > 1e-10:
            new_state = new_state / norm
            self.state = QuantumState(new_state, self.num_qubits)
        else:
            raise RuntimeError("State collapsed to zero - measurement probability was zero")
        
        return result
    
    def get_state(self) -> QuantumState:
        """Get the current quantum state."""
        return self.state
    
    def get_probabilities(self) -> np.ndarray:
        """Get measurement probabilities for each basis state."""
        return self.state.probabilities()

    # ------------------------------------------------------------------
    # User-friendly gate helpers
    # ------------------------------------------------------------------

    def _import_gates(self):
        from .gates import (
            h, x, y, z,
            s, t,
            phase, rotation_x, rotation_y, rotation_z,
            cnot, cz, swap, cphase
        )
        return {
            "h": h,
            "x": x,
            "y": y,
            "z": z,
            "s": s,
            "t": t,
            "phase": phase,
            "rotation_x": rotation_x,
            "rotation_y": rotation_y,
            "rotation_z": rotation_z,
            "cnot": cnot,
            "cz": cz,
            "swap": swap,
            "cphase": cphase,
        }

    def h(self, qubit: int) -> "_QuantumRegister":
        """Apply a Hadamard gate to ``qubit`` and return ``self``."""
        gates = self._import_gates()
        self.apply_single_qubit_gate(gates["h"](), qubit)
        return self

    def x(self, qubit: int) -> "_QuantumRegister":
        """Apply a Pauli-X gate to ``qubit`` and return ``self``."""
        gates = self._import_gates()
        self.apply_single_qubit_gate(gates["x"](), qubit)
        return self

    def y(self, qubit: int) -> "_QuantumRegister":
        """Apply a Pauli-Y gate to ``qubit`` and return ``self``."""
        gates = self._import_gates()
        self.apply_single_qubit_gate(gates["y"](), qubit)
        return self

    def z(self, qubit: int) -> "_QuantumRegister":
        """Apply a Pauli-Z gate to ``qubit`` and return ``self``."""
        gates = self._import_gates()
        self.apply_single_qubit_gate(gates["z"](), qubit)
        return self

    def s(self, qubit: int) -> "_QuantumRegister":
        """Apply an S-phase gate to ``qubit`` and return ``self``."""
        gates = self._import_gates()
        self.apply_single_qubit_gate(gates["s"](), qubit)
        return self

    def t(self, qubit: int) -> "_QuantumRegister":
        """Apply a T gate to ``qubit`` and return ``self``."""
        gates = self._import_gates()
        self.apply_single_qubit_gate(gates["t"](), qubit)
        return self

    def phase(self, qubit: int, phi: float) -> "_QuantumRegister":
        """Apply an arbitrary phase rotation to ``qubit``."""
        gates = self._import_gates()
        self.apply_single_qubit_gate(gates["phase"](phi), qubit)
        return self

    def rx(self, qubit: int, theta: float) -> "_QuantumRegister":
        """Apply an ``Rx(theta)`` rotation and return ``self``."""
        gates = self._import_gates()
        self.apply_single_qubit_gate(gates["rotation_x"](theta), qubit)
        return self

    def ry(self, qubit: int, theta: float) -> "_QuantumRegister":
        """Apply an ``Ry(theta)`` rotation and return ``self``."""
        gates = self._import_gates()
        self.apply_single_qubit_gate(gates["rotation_y"](theta), qubit)
        return self

    def rz(self, qubit: int, theta: float) -> "_QuantumRegister":
        """Apply an ``Rz(theta)`` rotation and return ``self``."""
        gates = self._import_gates()
        self.apply_single_qubit_gate(gates["rotation_z"](theta), qubit)
        return self

    def cnot(self, control: int, target: int) -> "_QuantumRegister":
        """Apply a CNOT gate with ``control`` and ``target`` qubits."""
        gates = self._import_gates()
        self.apply_two_qubit_gate(gates["cnot"](), control, target)
        return self

    def cz(self, control: int, target: int) -> "_QuantumRegister":
        """Apply a controlled-Z gate."""
        gates = self._import_gates()
        self.apply_two_qubit_gate(gates["cz"](), control, target)
        return self

    def swap(self, qubit1: int, qubit2: int) -> "_QuantumRegister":
        """Swap two qubits and return ``self``."""
        gates = self._import_gates()
        self.apply_two_qubit_gate(gates["swap"](), qubit1, qubit2)
        return self

    def cphase(self, control: int, target: int, phi: float) -> "_QuantumRegister":
        """Apply a controlled-phase gate with angle ``phi``."""
        gates = self._import_gates()
        self.apply_two_qubit_gate(gates["cphase"](phi), control, target)
        return self
    
    def __repr__(self):
        return f"_QuantumRegister(num_qubits={self.num_qubits})"
    
    def __str__(self):
        return f"QuantumRegister({self.num_qubits} qubits):\n  {self.state.to_string()}"
