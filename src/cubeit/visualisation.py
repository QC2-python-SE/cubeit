"""
Visualization and utility functions for quantum states and circuits.
"""

import numpy as np
from .register import _QuantumRegister as QuantumRegister, QuantumState
from .DMs import DensityMatrix2Qubit as DM
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D


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


def partial_trace(rho, keep, dims):
    """
    Perform partial trace on a density matrix.
    Args:
        rho: density matrix to trace
        keep: list of indices to keep, e.g. [0, 2] to keep subsystems 0 and 2
        dims: list of dimensions for each subsystem, e.g. [2, 2, 2] for three qubits
        Returns:
            reduced density matrix after tracing out unwanted subsystems
    """
    N = len(dims)
    reshaped = rho.reshape(dims + dims) # for a two qubit system this will reshape from (4,4) to (2,2,2,2)
    traced = reshaped
    for i in reversed(range(N)): # looping backwards avoids messing up the axis indices
        if i not in keep:
            traced = np.trace(traced, axis1=i, axis2=i+N)
    return traced

def state_to_reduced_density_matrix(system: QuantumRegister):
    """
    Convert a statvector into a list of reduced density matrices.
    Args:
        system: an instance of the class QuantumRegister
    Returns:
        rhos: list of reduced density matrices
    """
    if isinstance(system, QuantumRegister):
        state_vector = system.state.state
        rho = np.outer(state_vector, np.conj(state_vector))
        num_qubits = system.num_qubits
    elif isinstance(system, DM):
        rho = system.rho
        num_qubits = 2
    else:
        raise ValueError('Input must be a QuantumRegister or Density Matrix instance')
    dims = [2] * num_qubits
    rhos = []
    for i in range(num_qubits):
        rho_i = partial_trace(rho, [i], dims)
        rhos.append(rho_i)
    return rhos    

def bloch_from_density(rho):
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

def plot_bloch_sphere(system):
    """
    Plot the Bloch sphere representation of each qubit in the QuantumRegister or Density Matrix.
    
    Args:
        system: QuantumRegister or Density Matrix containing the qubits to visualize
    """

    if isinstance(system, QuantumRegister):
        rhos = state_to_reduced_density_matrix(system)
        state_vectors = []
        for rho in rhos:
            state_vectors.append(bloch_from_density(rho))
        num_qubits = system.num_qubits

    elif isinstance(system, DM):
        num_qubits = 2
        state_vectors = []
        rhos = state_to_reduced_density_matrix(system)
        for rho in rhos:
            state_vectors.append(bloch_from_density(rho))      

    else:
        raise ValueError('Input must be a QuantumRegister or Density Matrix instance')
    
    fig = plt.figure(figsize=(5 * num_qubits, 2.9))
    for i in range(num_qubits):
        ax = fig.add_subplot(1, num_qubits, i + 1, projection='3d')
        x,y,z = state_vectors[i]
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
        # add marker at the Bloch sphere centre
        ax.scatter([0], [0], [0], color='#D62728', s=90, zorder=10)

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

        # title and viewing angle for consistent perspective
        ax.set_title(f'Qubit {i}', fontsize=12, fontweight='semibold', pad=0)
        ax.view_init(elev=20, azim=30)

        # Add labels for |0> at the north pole (z=1) and |1> at the south pole (z=-1)
        # Slightly offset in z so the text doesn't overlap the sphere surface
        try:
            ax.text(0.0, 0.0, 1.08, '|0⟩', fontsize=12, ha='center', va='bottom')
            ax.text(0.0, 0.0, -1.1, '|1⟩', fontsize=12, ha='center', va='top')
        except Exception:
            # Some matplotlib/MPL backends may raise for 3D text; ignore failures
            pass

    fig.tight_layout()
    plt.show()
    return fig

def plot_circuit(system: QuantumRegister):
    """
    Plot a simple circuit diagram based on `system.history`.

    History format (as implemented in `register.py`):
      - single-qubit gates: [qubit_index, 'gate_name']
      - two-qubit gates: [qubit1_index, qubit2_index, 'gate_name']

    This draws horizontal wires for each qubit and places gates sequentially
    from left to right.
    """
    if isinstance(system, QuantumRegister):
        history = getattr(system, 'history', None)
        n = system.num_qubits
        if history is None:
            raise ValueError('System has no history attribute')
    elif isinstance(system, DM):
        DM_history = getattr(system, 'history', None)
        n = 2
        history = []
        for item in DM_history:
            name = item["gate"]
            qubit = item["target"]
            if isinstance(qubit, list) and len(qubit) == 2:
                history.append([qubit[0], qubit[1], name])
            else:
                history.append([qubit, name])
        if history is None:
            raise ValueError('System has no history attribute')
    if n < 1:
        raise ValueError('Number of qubits must be >= 1')

    # Layout params (tweaked for aesthetics)
    spacing_x = 1.4
    gate_w = 0.64
    gate_h = 0.5
    left_margin = 0.8
    top_margin = 0.2

    fig_width = max(6, left_margin + len(history) * spacing_x + 1.0)
    fig_height = max(1.6 + n * 0.9, 2.4)

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # y positions: qubit 0 at top (y = 0), qubit n-1 at bottom
    ys = [-(i) for i in range(n)]

    # Draw wires (low z-order so gates appear above them)
    wire_xstart = left_margin - 0.1
    wire_xend = left_margin + len(history) * spacing_x + 0.6
    for i, y in enumerate(ys):
        ax.hlines(y, wire_xstart, wire_xend, color='k', linewidth=1, zorder=0)
        # show initial |0> ket at the start of each wire
        ax.text(left_margin - 0.35, y, '|0⟩', fontsize=12, ha='right', va='center', zorder=3)

    # Choose colors from the same coolwarm colormap used by the Bloch sphere
    cmap = plt.cm.coolwarm
    single_color = cmap(0.2)
    two_color = cmap(0.8)
    neutral_color = cmap(0.5)

    # Draw gates along x axis
    for idx, item in enumerate(history):
        x = left_margin + idx * spacing_x + 0.5

        # single-qubit gate [qubit, 'name']
        if isinstance(item, list) and len(item) == 2:
            qubit, name = item
            if qubit < 0 or qubit >= n:
                continue
            y = ys[qubit]
            # rounded box for gate (drawn above wires)
            box = FancyBboxPatch((x - gate_w/2, y - gate_h/2), gate_w, gate_h,
                                 boxstyle="round,pad=0.02,rounding_size=0.08",
                                 linewidth=1, edgecolor='#222222', facecolor=single_color, zorder=5)
            ax.add_patch(box)
            ax.text(x, y, str(name).upper(), fontsize=10, ha='center', va='center', zorder=6, color='#111111', weight='semibold')

        # two-qubit gate [q1, q2, 'name']
        elif isinstance(item, list) and len(item) == 3:
            q1, q2, name = item
            # determine control and target y positions
            if q1 < 0 or q1 >= n or q2 < 0 or q2 >= n:
                continue
            y1 = ys[q1]
            y2 = ys[q2]
            # vertical line connecting (drawn behind gate symbols)
            ax.vlines(x, min(y1, y2), max(y1, y2), color='k', linewidth=1, zorder=1)

            # determine control/target y positions
            control_y = y1
            target_y = y2
            gate_name = str(name).lower()

            # For SWAP: draw X symbols on both wires and do not draw a control dot
            if gate_name == 'swap':
                ax.text(x, y1, 'X', fontsize=16, ha='center', va='center', zorder=6, color='#111111')
                ax.text(x, y2, 'X', fontsize=16, ha='center', va='center', zorder=6, color='#111111')
                # midpoint connector for visual continuity
                mid = (y1 + y2) / 2
                ax.add_patch(Circle((x, mid), 0.003, color='#444444', zorder=3))
            else:
                # draw control dot for controlled-like gates
                ax.add_patch(Circle((x, control_y), 0.06, color='k', zorder=5))

                # target depiction depends on gate name
                if gate_name == 'cnot' or gate_name == 'cx':
                    # target: circle with plus
                    circ = Circle((x, target_y), 0.12, fill=False, ec='k', linewidth=1.2, zorder=5)
                    ax.add_patch(circ)
                    ax.text(x, target_y, '+', fontsize=10, ha='center', va='center', zorder=6)
                elif gate_name in ('cz', 'cphase'):
                    # target: box with Z/P
                    rect = Rectangle((x - gate_w/2, target_y - gate_h/2), gate_w, gate_h, fill=True, color=two_color, ec='k', zorder=5)
                    ax.add_patch(rect)
                    lbl = 'Z' if gate_name == 'cz' else 'P'
                    ax.text(x, target_y, lbl, fontsize=9, ha='center', va='center', zorder=6)
                else:
                    # generic two-qubit box at midpoint
                    mid = (target_y + control_y) / 2
                    rect = Rectangle((x - gate_w/2, mid - gate_h/2), gate_w, gate_h, fill=True, color=neutral_color, ec='k', zorder=5)
                    ax.add_patch(rect)
                    ax.text(x, mid, str(name).upper(), fontsize=9, ha='center', va='center', zorder=6)

        else:
            # unknown history entry, just print it as text at this x
            try:
                txt = str(item).upper()
            except Exception:
                txt = str(item)
            ax.text(x, 0, txt, fontsize=8, ha='center', va='center', zorder=6)

    # Legend for non-experts (placed to the right)
    try:
        legend_elements = [
            FancyBboxPatch((0,0),1,1, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor=single_color, edgecolor='#222222'),
            Circle((0,0), 0.06, color='#111111'),
            Line2D([0],[0], color='#222222', lw=1),
            Circle((0,0), 0.12, fill=False, ec='#111111', linewidth=1.2),
            Line2D([0],[0], marker='x', color='w', markeredgecolor='#111111', markersize=10, linestyle=''),
            Rectangle((0,0),1,1, facecolor=two_color, edgecolor='#222222')
        ]
        legend_labels = [
            'Single-qubit gate',
            'Control dot (control qubit)',
            'Wire (timeline)',
            'Target (CNOT)',
            'SWAP (X on both wires)',
            'Controlled-phase / Z box'
        ]
        ax.legend(legend_elements, legend_labels, loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False)
    except Exception:
        pass

    # Cosmetic
    ax.set_ylim(min(ys) - 0.8, max(ys) + 0.8)
    ax.set_xlim(0, left_margin + len(history) * spacing_x + 0.4)
    # Keep equal aspect so circles appear circular (not squashed)
    try:
        ax.set_aspect('equal', adjustable='box')
    except Exception:
        pass
    ax.axis('off')
    fig.tight_layout()
    plt.show()
    return fig
