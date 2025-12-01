"""
Universal gate set and additional quantum gates for two-qubit systems.
!! This file is identical to gates.py except that it also returns the gate names, intended for use with the density matrix class. !!

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
        np.ndarray: 2x2 Hadamard gate matrix
        str: Gate name
    """
    mat = (1/np.sqrt(2)) * np.array([
        [1,  1],
        [1, -1]
    ], dtype=complex)
    name = "Had"
    return mat, name

def s() -> np.ndarray:
    """
    Phase gate (S gate) - applies π/2 phase.
    
    S = [[1, 0],
         [0, i]]
    
    Returns:
        2x2 Phase gate matrix
    """
    mat = np.array([
        [1, 0],
        [0, 1j]
    ], dtype=complex)
    name = "S"
    return mat, name

def t() -> np.ndarray:
    """
    T gate (π/8 gate) - applies π/4 phase.
    
    T = [[1, 0],
         [0, exp(iπ/4)]]
    
    Returns:
        np.ndarray: 2x2 T gate matrix
        str: Gate name
    """
    mat = np.array([
        [1, 0],
        [0, np.exp(1j * np.pi / 4)]
    ], dtype=complex)
    name = "T"
    return mat, name

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
        np.ndarray: 4x4 CNOT gate matrix (control=0, target=1)
        str: Gate name
    """
    mat = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=complex)
    name = "CNOT"
    return mat, name

def cnot_10() -> np.ndarray:
    """
    CNOT gate with control on qubit 1, target on qubit 0.
    
    Returns:
        np.ndarray: 4x4 CNOT gate matrix (control=1, target=0)
        str: Gate name
    """
    mat = np.array([
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0]
    ], dtype=complex)
    name = "CNOT_10"
    return mat, name

# ============================================================================
# Pauli Gates
# ============================================================================

def x() -> np.ndarray:
    """
    Pauli-X gate (bit flip / NOT gate).
    
    X = [[0, 1],
         [1, 0]]
    
    Returns:
        np.ndarray: 2x2 Pauli-X gate matrix
        str: Gate name    
    """
    mat = np.array([
        [0, 1],
        [1, 0]
    ], dtype=complex)
    name = "X"
    return mat, name

def y() -> np.ndarray:
    """
    Pauli-Y gate.
    
    Y = [[0, -i],
         [i,  0]]
    
    Returns:
        np.ndarray: 2x2 Pauli-Y gate matrix
        str: Gate name
    """
    mat = np.array([
        [0, -1j],
        [1j, 0]
    ], dtype=complex)
    name = "Y"
    return mat, name

def z() -> np.ndarray:
    """
    Pauli-Z gate (phase flip).
    
    Z = [[1,  0],
         [0, -1]]
    
    Returns:
        np.ndarray: 2x2 Pauli-Z gate matrix
        str: Gate name
    """
    mat = np.array([
        [1,  0],
        [0, -1]
    ], dtype=complex)
    name = "Z"
    return mat, name

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
        np.ndarray: 2x2 Phase gate matrix
        str: Gate name
    """
    mat = np.array([
        [1, 0],
        [0, np.exp(1j * phi)]
    ], dtype=complex)
    name = f"Phase({phi:.2f})"
    return mat, name


def rotation_x(theta: float) -> np.ndarray:
    """
    Rotation around X-axis.
    
    Rx(θ) = [[cos(θ/2), -i*sin(θ/2)],
             [-i*sin(θ/2), cos(θ/2)]]
    
    Args:
        theta: Rotation angle in radians
    
    Returns:
        np.ndarray: 2x2 Rotation-X gate matrix
        str: Gate name
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    mat = np.array([
        [c, -1j * s],
        [-1j * s, c]
    ], dtype=complex)
    name = f"Rx({theta:.2f})"
    return mat, name

def rotation_y(theta: float) -> np.ndarray:
    """
    Rotation around Y-axis.
    
    Ry(θ) = [[cos(θ/2), -sin(θ/2)],
             [sin(θ/2),  cos(θ/2)]]
    
    Args:
        theta: Rotation angle in radians
    
    Returns:
        np.ndarray: 2x2 Rotation-Y gate matrix
        str: Gate name
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    mat = np.array([
        [c, -s],
        [s, c]
    ], dtype=complex)
    name = f"Ry({theta:.2f})"
    return mat, name

def rotation_z(theta: float) -> np.ndarray:
    """
    Rotation around Z-axis.
    
    Rz(θ) = [[exp(-iθ/2), 0],
             [0, exp(iθ/2)]]
    
    Args:
        theta: Rotation angle in radians
    
    Returns:
        np.ndarray: 2x2 Rotation-Z gate matrix
        str: Gate name
    """
    mat = np.array([
        [np.exp(-1j * theta / 2), 0],
        [0, np.exp(1j * theta / 2)]
    ], dtype=complex)
    name = f"Rz({theta:.2f})"
    return mat, name

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
        np.ndarray: 4x4 SWAP gate matrix
        str: Gate name
    """
    mat = np.array([
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ], dtype=complex)
    name = "SWAP"
    return mat, name

def cz() -> np.ndarray:
    """
    Controlled-Z gate.
    
    CZ = [[1, 0, 0,  0],
          [0, 1, 0,  0],
          [0, 0, 1,  0],
          [0, 0, 0, -1]]
    
    Returns:
        np.ndarray: 4x4 Controlled-Z gate matrix
        str: Gate name
    """
    mat = np.array([
        [1, 0, 0,  0],
        [0, 1, 0,  0],
        [0, 0, 1,  0],
        [0, 0, 0, -1]
    ], dtype=complex)
    name = "CZ"
    return mat, name

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
        np.ndarray: 4x4 Controlled-Phase gate matrix
        str: Gate name
    """
    mat = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, np.exp(1j * phi)]
    ], dtype=complex)
    name = f"CPHASE({phi:.2f})"
    return mat, name