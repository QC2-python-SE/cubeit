"""
Universal gate set and additional quantum gates for two-qubit systems.

Universal gate set: {H, S, T, CNOT}
This set can approximate any unitary operation to arbitrary precision.
"""

import numpy as np
from typing import Callable


# ============================================================================
# Universal Gate Set
# ============================================================================

def h() -> np.ndarray:
    """
    Hadamard gate - creates superposition.
    
    H = (1/√2) * [[1,  1],
                  [1, -1]]
    
    Returns:
        2x2 Hadamard gate matrix
    """
    return (1/np.sqrt(2)) * np.array([
        [1,  1],
        [1, -1]
    ], dtype=complex)


def s() -> np.ndarray:
    """
    Phase gate (S gate) - applies π/2 phase.
    
    S = [[1, 0],
         [0, i]]
    
    Returns:
        2x2 Phase gate matrix
    """
    return np.array([
        [1, 0],
        [0, 1j]
    ], dtype=complex)


def t() -> np.ndarray:
    """
    T gate (π/8 gate) - applies π/4 phase.
    
    T = [[1, 0],
         [0, exp(iπ/4)]]
    
    Returns:
        2x2 T gate matrix
    """
    return np.array([
        [1, 0],
        [0, np.exp(1j * np.pi / 4)]
    ], dtype=complex)


def cnot() -> np.ndarray:
    """
    Controlled-NOT gate (two-qubit gate).
    
    CNOT = [[1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]]
    
    Controls on qubit 0, targets qubit 1.
    For control on qubit 1, use CNOT_10().
    
    Returns:
        4x4 CNOT gate matrix (control=0, target=1)
    """
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)


def cnot_10() -> np.ndarray:
    """
    CNOT gate with control on qubit 1, target on qubit 0.
    
    Returns:
        4x4 CNOT gate matrix (control=1, target=0)
    """
    return np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0]
    ], dtype=complex)


# ============================================================================
# Pauli Gates
# ============================================================================

def x() -> np.ndarray:
    """
    Pauli-X gate (bit flip / NOT gate).
    
    X = [[0, 1],
         [1, 0]]
    
    Returns:
        2x2 Pauli-X gate matrix
    """
    return np.array([
        [0, 1],
        [1, 0]
    ], dtype=complex)


def y() -> np.ndarray:
    """
    Pauli-Y gate.
    
    Y = [[0, -i],
         [i,  0]]
    
    Returns:
        2x2 Pauli-Y gate matrix
    """
    return np.array([
        [0, -1j],
        [1j, 0]
    ], dtype=complex)


def z() -> np.ndarray:
    """
    Pauli-Z gate (phase flip).
    
    Z = [[1,  0],
         [0, -1]]
    
    Returns:
        2x2 Pauli-Z gate matrix
    """
    return np.array([
        [1,  0],
        [0, -1]
    ], dtype=complex)


# ============================================================================
# Parameterized Gates
# ============================================================================

def phase(phi: float) -> np.ndarray:
    """
    Phase gate with arbitrary phase.
    
    Phase(φ) = [[1, 0],
                [0, exp(iφ)]]
    
    Args:
        phi: Phase angle in radians
    
    Returns:
        2x2 Phase gate matrix
    """
    return np.array([
        [1, 0],
        [0, np.exp(1j * phi)]
    ], dtype=complex)


def rotation_x(theta: float) -> np.ndarray:
    """
    Rotation around X-axis.
    
    Rx(θ) = [[cos(θ/2), -i*sin(θ/2)],
             [-i*sin(θ/2), cos(θ/2)]]
    
    Args:
        theta: Rotation angle in radians
    
    Returns:
        2x2 Rotation-X gate matrix
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([
        [c, -1j * s],
        [-1j * s, c]
    ], dtype=complex)


def rotation_y(theta: float) -> np.ndarray:
    """
    Rotation around Y-axis.
    
    Ry(θ) = [[cos(θ/2), -sin(θ/2)],
             [sin(θ/2),  cos(θ/2)]]
    
    Args:
        theta: Rotation angle in radians
    
    Returns:
        2x2 Rotation-Y gate matrix
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([
        [c, -s],
        [s, c]
    ], dtype=complex)


def rotation_z(theta: float) -> np.ndarray:
    """
    Rotation around Z-axis.
    
    Rz(θ) = [[exp(-iθ/2), 0],
             [0, exp(iθ/2)]]
    
    Args:
        theta: Rotation angle in radians
    
    Returns:
        2x2 Rotation-Z gate matrix
    """
    return np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=complex)


# ============================================================================
# Additional Two-Qubit Gates
# ============================================================================

def swap() -> np.ndarray:
    """
    SWAP gate - exchanges two qubits.
    
    SWAP = [[1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]]
    
    Returns:
        4x4 SWAP gate matrix
    """
    return np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=complex)


def cz() -> np.ndarray:
    """
    Controlled-Z gate.
    
    CZ = [[1, 0, 0,  0],
          [0, 1, 0,  0],
          [0, 0, 1,  0],
          [0, 0, 0, -1]]
    
    Returns:
        4x4 Controlled-Z gate matrix
    """
    return np.array([
        [1, 0, 0,  0],
        [0, 1, 0,  0],
        [0, 0, 1,  0],
        [0, 0, 0, -1]
    ], dtype=complex)


def cphase(phi: float) -> np.ndarray:
    """
    Controlled-Phase gate with arbitrary phase.
    
    CPHASE(φ) = [[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 1, 0],
                 [0, 0, 0, exp(iφ)]]
    
    Args:
        phi: Phase angle in radians
    
    Returns:
        4x4 Controlled-Phase gate matrix
    """
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, np.exp(1j * phi)]
    ], dtype=complex)

