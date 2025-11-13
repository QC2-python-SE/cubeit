"""
Visualization and utility functions for quantum states and circuits.
"""

import numpy as np
from typing import List, Tuple
from .register import _QuantumRegister as QuantumRegister, QuantumState


def print_state(system: QuantumRegister):
    """Print the current quantum state in a readable format."""
    print(system.get_state().to_string())


def print_probabilities(system: QuantumRegister):
    """Print measurement probabilities for each basis state."""
    probs = system.get_probabilities()
    num_qubits = system.num_qubits
    
    print("Measurement Probabilities:")
    for i, prob in enumerate(probs):
        if prob > 1e-10:  # Only print non-zero probabilities
            binary = format(i, f'0{num_qubits}b')
            basis_state = f"|{binary}⟩"
            print(f"  {basis_state}: {prob:.4f} ({prob*100:.2f}%)")


def simulate_measurements(system: QuantumRegister, num_samples: int = 1000) -> dict:
    """
    Simulate multiple measurements and return statistics.
    
    Args:
        system: Two-qubit system to measure
        num_samples: Number of measurements to perform
    
    Returns:
        Dictionary with measurement counts
    """
    counts = {}
    
    # Create a copy to avoid modifying the original system
    state = system.get_state()
    
    for _ in range(num_samples):
        # Create a temporary system with the same state
        temp_system = QuantumRegister(system.num_qubits, state)
        result = temp_system.measure()
        key = "".join(str(bit) for bit in result)
        counts[key] = counts.get(key, 0) + 1
    
    return counts


def print_measurement_stats(system: QuantumRegister, num_samples: int = 1000):
    """
    Print measurement statistics from multiple simulations.
    
    Args:
        system: Two-qubit system to measure
        num_samples: Number of measurements to perform
    """
    counts = simulate_measurements(system, num_samples)
    
    print(f"\nMeasurement Statistics ({num_samples} samples):")
    for state, count in counts.items():
        percentage = (count / num_samples) * 100
        print(f"  |{state}⟩: {count:4d} ({percentage:5.2f}%)")


def fidelity(system1: QuantumRegister, system2: QuantumRegister) -> float:
    """
    Calculate fidelity between two quantum states.
    
    Args:
        system1: First two-qubit system
        system2: Second two-qubit system
    
    Returns:
        Fidelity value between 0 and 1
    """
    return system1.get_state().fidelity(system2.get_state())

# write function take cubeit QuantumRegister and plot the bloch sphere representation of each qubit in the register
# write von neumann entropy function to show entanglement
# do some tests