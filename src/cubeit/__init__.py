"""
CubeIt - A quantum software package for n-qubit systems with universal gate set.

This package provides a simple and intuitive interface for working with n-qubit
quantum systems, including a universal gate set for quantum computation.
"""

from .register import QuantumState, _QuantumRegister
from .gates import (
    had, s, t, cnot, cnot_10,
    x, y, z,
    phase, rotation_x, rotation_y, rotation_z,
    swap, cz, cphase
)
from typing import Optional

__version__ = "0.2.0"
__all__ = [
    "QuantumState",
    "had", "s", "t", "cnot", "cnot_10",
    "x", "y", "z",
    "phase", "rotation_x", "rotation_y", "rotation_z",
    "swap", "cz", "cphase",
    "quantumregister",
    "get_state",
    "measure",
]


def quantumregister(num_qubits: int = 2, initial_state=None) -> _QuantumRegister:
    """Convenience factory that mirrors :class:`QuantumRegister`."""
    return _QuantumRegister(num_qubits, initial_state)


def get_state(
    register: _QuantumRegister,
    *,
    precision: int = 3,
    print_state: bool = True,
    as_array: bool = False,
) -> str:
    """
    Print (optionally) and return a human-readable representation of the state.

    Args:
        register: The quantum register to inspect.
        precision: Decimal precision for amplitudes.
        print_state: Whether to print the formatted state string.
        as_array: If True, also return the underlying numpy array copy.

    Returns:
        State description string, or numpy array copy when ``as_array`` is True.
    """

    state = register.get_state()
    description = state.to_string(precision=precision)
    if print_state:
        print(description)
    if as_array:
        return state.state.copy()
    return description


def measure(
    register: _QuantumRegister,
    *,
    qubit: Optional[int] = None,
    print_result: bool = True,
):
    """Measure an entire register or a single qubit.

    Args:
        register: The quantum register to measure.
        qubit: Optional index to measure a single qubit.
        print_result: Whether to print the measurement result.

    Returns:
        Measurement outcome: ``0``/``1`` for single-qubit results, or a tuple for
        multi-qubit measurements when ``qubit`` is ``None``.
    """

    if qubit is None:
        result = register.measure()
        if isinstance(result, tuple) and len(result) == 1:
            result = result[0]
    else:
        result = register.measure_qubit(qubit)

    if print_result:
        print(result)
    return result

