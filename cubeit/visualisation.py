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
    elems = list(system.state.state)
    for idx in range(0, len(elems), 2):
        if idx + 1 < len(elems):
            state_vectors.append([elems[idx], elems[idx + 1]])
        else:
            # if odd length, pair last element with 0
            print("Odd number of elements in state vector, pairing last element with 0.")
            state_vectors.append([elems[idx], 0+0j])

    for i in range(num_qubits):
        ax = fig.add_subplot(1, num_qubits, i + 1, projection='3d')
        a, b = state_vectors[i]
        x = 2 * np.real(a * np.conj(b))
        y = 2 * np.imag(a * np.conj(b))
        z = abs(a)**2 - abs(b)**2

        # Draw a softly shaded Bloch sphere
        u = np.linspace(0, 2 * np.pi, 60)
        v = np.linspace(0, np.pi, 30)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones_like(u), np.cos(v))

        # subtle colormap shading for better depth perception
        cmap = plt.cm.coolwarm
        face_colors = cmap((zs + 1) / 2)
        ax.plot_surface(xs, ys, zs, facecolors=face_colors, rstride=1, cstride=1,
                linewidth=0, antialiased=True, alpha=0.18, shade=True)

        # draw faint equator and meridian lines for reference
        ax.plot(np.cos(u), np.sin(u), 0, color='gray', lw=0.8, alpha=0.6)
        ax.plot(np.cos(u), np.zeros_like(u), np.sin(u), color='gray', lw=0.6, alpha=0.45)
        ax.plot(np.zeros_like(u), np.cos(u), np.sin(u), color='gray', lw=0.6, alpha=0.45)

        # normalize and draw state vector as an arrow, plus a bright marker at the tip
        vec = np.array([x, y, z], dtype=float)
        r = np.linalg.norm(vec)
        if r < 1e-12:
            vec_display = np.array([0.0, 0.0, 0.0])
        else:
            vec_display = vec / r  # direction
        # scale arrow to actual radius r so pure states sit on sphere surface
        arrow_length = r if r > 1e-12 else 0.0
        ax.quiver(0, 0, 0, vec_display[0], vec_display[1], vec_display[2],
              length=arrow_length, color='#D62728', linewidth=2.0,
              arrow_length_ratio=0.12, normalize=False)

        # bright marker for the state and subtle shadow for depth
        ax.scatter([vec_display[0] * arrow_length], [vec_display[1] * arrow_length],
               [vec_display[2] * arrow_length], color='#D62728', s=90, edgecolor='k', zorder=10)

        # nice axes: show only -1, 0, 1 ticks and label them
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xticks([-1.0, 0.0, 1.0])
        ax.set_yticks([-1.0, 0.0, 1.0])
        ax.set_zticks([-1.0, 0.0, 1.0])
        ax.set_xlabel('X', fontsize=10)
        ax.set_ylabel('Y', fontsize=10)
        ax.set_zlabel('Z', fontsize=10)

        # clean visual style: remove grid/pane colors and set aspect
        ax.grid(False)
        try:
            ax.xaxis.pane.fill = False
            ax.yaxis.pane.fill = False
            ax.zaxis.pane.fill = False
        except Exception:
            pass
        # Matplotlib 3.3+ way to ensure equal aspect ratio for 3D
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass

        # title and viewing angle for consistent, pleasing perspective
        ax.set_title(f'Qubit {i}', fontsize=12, fontweight='semibold', pad=10)
        ax.view_init(elev=20, azim=30)

    plt.tight_layout()
    plt.show()