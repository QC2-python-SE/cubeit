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

def plot_bloch_sphere(system: QuantumRegister):
    """
    Plot the Bloch sphere representation of each qubit in the QuantumRegister.
    
    Args:
        system: QuantumRegister containing the qubits to visualize
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    num_qubits = system.num_qubits
    fig = plt.figure(figsize=(5 * num_qubits, 5))
    state_vectors = []
    for i in system.state.state:
        state_vectors.append(i)

    for i in range(num_qubits):
        ax = fig.add_subplot(1, num_qubits, i + 1, projection='3d')
        x, y, z = state_vectors[i].real, state_vectors[i].imag, 0

        # Draw Bloch sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones(np.size(u)), np.cos(v))
        ax.plot_surface(xs, ys, zs, color='c', alpha=0.1)

        # Draw state vector
        ax.quiver(0, 0, 0, x, y, z, color='r', linewidth=2)
        ax.set_title(f'Qubit {i}')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

    plt.show()