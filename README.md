# CubeIt

A quantum software package for two-qubit systems with a universal gate set.

CubeIt provides a simple and intuitive interface for working with two-qubit quantum systems, including a universal gate set that can approximate any quantum operation to arbitrary precision.

## Features

- **Universal Gate Set**: {H, S, T, CNOT} - can approximate any unitary operation
- **Two-Qubit System**: Full support for two-qubit quantum states and operations
- **Additional Gates**: Pauli gates (X, Y, Z), rotations, and two-qubit gates
- **Measurement Operations**: Full and partial qubit measurements
- **Visualization Tools**: State visualization and measurement statistics
- **Bell States**: Easy creation of maximally entangled states

## Installation

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

## Quick Start

### Basic Usage

```python
from cubeit import TwoQubitSystem, H, CNOT

# Create a two-qubit system (starts in |00⟩ state)
system = TwoQubitSystem()

# Apply Hadamard gate to qubit 0
system.apply_single_qubit_gate(H(), 0)

# Apply CNOT gate (control=0, target=1)
system.apply_gate(CNOT())

# View the state
print(system)
# Output: TwoQubitSystem:
#   0.707|00⟩ + 0.707|11⟩

# Measure the qubits
result = system.measure()
print(f"Measurement result: {result}")
```

### Universal Gate Set

The universal gate set consists of:

- **H (Hadamard)**: Creates superposition
- **S (Phase)**: Applies π/2 phase
- **T (π/8 gate)**: Applies π/4 phase
- **CNOT**: Controlled-NOT gate (two-qubit)

```python
from cubeit import TwoQubitSystem, H, S, T, CNOT

system = TwoQubitSystem()

# Apply universal gates
system.apply_single_qubit_gate(H(), 0)  # Hadamard on qubit 0
system.apply_single_qubit_gate(S(), 0)   # Phase gate on qubit 0
system.apply_single_qubit_gate(T(), 1)   # T gate on qubit 1
system.apply_gate(CNOT())                # CNOT gate

print(system)
```

### Creating Bell States

```python
from cubeit.visualization import create_bell_state, print_state, print_measurement_stats

# Create a Bell state |Φ⁺⟩ = (|00⟩ + |11⟩) / √2
bell = create_bell_state("phi_plus")

print_state(bell)
# Output: 0.707|00⟩ + 0.707|11⟩

# Simulate measurements
print_measurement_stats(bell, num_samples=1000)
```

### Measurement Probabilities

```python
from cubeit.visualization import print_probabilities

system = TwoQubitSystem()
system.apply_single_qubit_gate(H(), 0)
system.apply_gate(CNOT())

print_probabilities(system)
# Output:
# Measurement Probabilities:
#   |00⟩: 0.5000 (50.00%)
#   |01⟩: 0.0000 (0.00%)
#   |10⟩: 0.0000 (0.00%)
#   |11⟩: 0.5000 (50.00%)
```

### Additional Gates

```python
from cubeit import TwoQubitSystem, X, Y, Z, RotationX, Phase

system = TwoQubitSystem()

# Pauli gates
system.apply_single_qubit_gate(X(), 0)  # Bit flip
system.apply_single_qubit_gate(Y(), 1)  # Pauli-Y
system.apply_single_qubit_gate(Z(), 0)  # Phase flip

# Parameterized gates
system.apply_single_qubit_gate(RotationX(np.pi/4), 0)  # Rotate around X-axis
system.apply_single_qubit_gate(Phase(np.pi/3), 1)      # Arbitrary phase
```

### Two-Qubit Gates

```python
from cubeit import TwoQubitSystem, CNOT, CNOT_10, SWAP, CZ

system = TwoQubitSystem()

# CNOT with control on qubit 0, target on qubit 1
system.apply_gate(CNOT())

# CNOT with control on qubit 1, target on qubit 0
system.apply_gate(CNOT_10())

# SWAP gate
system.apply_gate(SWAP())

# Controlled-Z gate
system.apply_gate(CZ())
```

### Partial Measurement

```python
system = TwoQubitSystem()
system.apply_single_qubit_gate(H(), 0)
system.apply_gate(CNOT())

# Measure only qubit 0 (collapses the state)
result = system.measure_qubit(0)
print(f"Qubit 0 measured: {result}")

# Now measure qubit 1
result = system.measure_qubit(1)
print(f"Qubit 1 measured: {result}")
```

## API Reference

### TwoQubitSystem

Main class for working with two-qubit quantum systems.

**Methods:**
- `apply_gate(gate_matrix)`: Apply a 4x4 gate matrix
- `apply_single_qubit_gate(gate_matrix, qubit)`: Apply a 2x2 gate to a specific qubit
- `measure()`: Measure both qubits
- `measure_qubit(qubit)`: Measure a single qubit (partial measurement)
- `get_state()`: Get the current quantum state
- `get_probabilities()`: Get measurement probabilities

### Gates

**Universal Gate Set:**
- `H()`: Hadamard gate
- `S()`: Phase gate
- `T()`: T gate (π/8 gate)
- `CNOT()`: Controlled-NOT gate (control=0, target=1)
- `CNOT_10()`: CNOT gate (control=1, target=0)

**Pauli Gates:**
- `X()`: Pauli-X (bit flip)
- `Y()`: Pauli-Y
- `Z()`: Pauli-Z (phase flip)

**Parameterized Gates:**
- `Phase(phi)`: Phase gate with arbitrary phase
- `RotationX(theta)`: Rotation around X-axis
- `RotationY(theta)`: Rotation around Y-axis
- `RotationZ(theta)`: Rotation around Z-axis

**Two-Qubit Gates:**
- `SWAP()`: SWAP gate
- `CZ()`: Controlled-Z gate
- `CPHASE(phi)`: Controlled-Phase gate

## Examples

### Example 1: Creating a Bell State

```python
from cubeit import TwoQubitSystem, H, CNOT

system = TwoQubitSystem()
system.apply_single_qubit_gate(H(), 0)
system.apply_gate(CNOT())

print(system)
# Output: TwoQubitSystem:
#   0.707|00⟩ + 0.707|11⟩
```

### Example 2: Quantum Teleportation Circuit

```python
from cubeit import TwoQubitSystem, H, CNOT, X, Z

# Initialize: qubit 0 is the state to teleport, qubits 1-2 are entangled
system = TwoQubitSystem()

# Create Bell pair between qubits 1 and 2
# (In a 2-qubit system, we'll use qubits 0 and 1)
system.apply_single_qubit_gate(H(), 0)
system.apply_gate(CNOT())

# Apply teleportation gates
system.apply_single_qubit_gate(H(), 0)
# ... (simplified example)
```

### Example 3: Measurement Statistics

```python
from cubeit import TwoQubitSystem, H, CNOT
from cubeit.visualization import print_measurement_stats

system = TwoQubitSystem()
system.apply_single_qubit_gate(H(), 0)
system.apply_gate(CNOT())

# Run 1000 measurements and see statistics
print_measurement_stats(system, num_samples=1000)
```

## Theory

### Universal Gate Set

The set {H, S, T, CNOT} forms a universal gate set for quantum computation. This means that any unitary operation on any number of qubits can be approximated to arbitrary precision using only these gates.

- **H**: Creates superposition states
- **S**: Applies π/2 phase rotation
- **T**: Applies π/4 phase rotation
- **CNOT**: Creates entanglement between qubits

### Two-Qubit States

A two-qubit system has a 4-dimensional state space with basis states:
- |00⟩ = [1, 0, 0, 0]ᵀ
- |01⟩ = [0, 1, 0, 0]ᵀ
- |10⟩ = [0, 0, 1, 0]ᵀ
- |11⟩ = [0, 0, 0, 1]ᵀ

Any two-qubit state can be written as:
|ψ⟩ = α|00⟩ + β|01⟩ + γ|10⟩ + δ|11⟩

where |α|² + |β|² + |γ|² + |δ|² = 1.

## Requirements

- Python >= 3.8
- NumPy >= 1.20.0

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
