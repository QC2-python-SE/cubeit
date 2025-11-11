# CubeIt

CubeIt is a lightweight quantum playground for **n-qubit** registers. Universal gate set, built-in measurement
helpers, and utilities for visualisation and testing.

## Features

- **N-Qubit Registers:** Create registers of any size with `quantumregister(n)`.
- **Universal Gate Set:** Lower-case factories (`h()`, `s()`, `t()`, `cnot()`, …)
  and ergonomic instance helpers (`qr.h(i)`, `qr.cnot(control, target)`).
- **State Introspection:** `get_state()` pretty prints amplitudes, `measure()`
  collapses qubits and returns classical outcomes.
- **Utility Modules:** Measurement statistics, Bell-state builders, fidelity
  checks, and more under `cubeit.visualization`.
- **Pytest-Friendly:** Extensive suite (see `example.py`) covering helpers,
  measurement logic, GHZ states, and unitarity tests.

## Installation

```bash
pip install -r requirements.txt
```

Or install as a package:

```bash
pip install -e .
```

## Quick Start

```python
from cubeit import quantumregister, get_state, measure

# 1) Create a 4-qubit register (|0000⟩)
qr = quantumregister(4)

# 2) Build a circuit with fluent helpers
qr.h(0)          # Hadamard on qubit 0
qr.cnot(0, 1)    # Entangle qubit 0 and 1
qr.rx(2, 0.5)    # Rotate qubit 2 around X by 0.5 radians
qr.cz(1, 3)      # Controlled-Z between qubit 1 and 3

# 3) Inspect the statevector (pretty printed)
get_state(qr)

# 4) Measure – collapses the state and returns a bitstring
result = measure(qr)
print("Measurement:", result)
```

### Gate Helpers at a Glance

| Helper | Description |
| ------ | ----------- |
| `qr.h(i)` | Hadamard on qubit *i* |
| `qr.x(i)`, `qr.y(i)`, `qr.z(i)` | Pauli gates |
| `qr.s(i)`, `qr.t(i)` | Phase / π/8 gates |
| `qr.rx(i, θ)`, `qr.ry(i, θ)`, `qr.rz(i, θ)` | Rotations |
| `qr.cnot(control, target)` | Controlled-NOT |
| `qr.cz(control, target)` | Controlled-Z |
| `qr.cphase(control, target, φ)` | Controlled-phase |
| `qr.swap(a, b)` | Swap two qubits |

Prefer something functional? Import the factories directly:

```python
from cubeit import h, cnot, quantumregister

qr = quantumregister(2)
qr.apply_single_qubit_gate(h(), 0)
qr.apply_two_qubit_gate(cnot(), 0, 1)
```

### Measurement & Probabilities

```python
from cubeit import quantumregister, get_state, measure
from cubeit.visualization import print_probabilities

qr = quantumregister(2).h(0).cnot(0, 1)

print_probabilities(qr)
# Measurement Probabilities:
#   |00⟩: 0.5000 (50.00%)
#   |11⟩: 0.5000 (50.00%)

measure(qr)  # collapses the register and prints the classical outcome
```

### Bell States & Visualisation

```python
from cubeit.visualization import create_bell_state, print_state, print_measurement_stats

bell = create_bell_state("phi_plus")
print_state(bell)                 # 0.707|00⟩ + 0.707|11⟩
print_measurement_stats(bell)     # Monte-Carlo sampling
```

## API Reference

### `quantumregister`

Factory returning an instance of the internal `_QuantumRegister`. Methods:

- `apply_gate(matrix)` / `apply_single_qubit_gate(matrix, qubit)`
- `measure()` & `measure_qubit(qubit)`
- `get_state()` → returns a `QuantumState`
- `get_probabilities()` → `np.ndarray`
- Fluent helpers: `h`, `x`, `y`, `z`, `s`, `t`, `phase`, `rx`, `ry`, `rz`,
  `cnot`, `cz`, `cphase`, `swap`

### Gate Factories

Import functions directly from `cubeit` when you need raw matrices:

```python
from cubeit import h, x, cnot, swap

u = h()          # 2x2 Hadamard matrix
cx = cnot()      # 4x4 CNOT (control=0, target=1)
swap_gate = swap()
```

## Examples

### Example 1: Creating a Bell State

```python
from cubeit import quantumregister

qr = quantumregister(2)
qr.h(0).cnot(0, 1)
print(qr)
# QuantumRegister(2 qubits):
#   0.707|00⟩ + 0.707|11⟩
```

### Example 2: Measurement Statistics

```python
from cubeit import quantumregister
from cubeit.visualization import print_measurement_stats

qr = quantumregister(2).h(0).cnot(0, 1)

print_measurement_stats(qr, num_samples=1000)
```

## Theory

### Universal Gate Set

The set {H, S, T, CNOT} (and consequently their lowercase counterparts) forms a
universal gate set. Any unitary operation can be approximated to arbitrary
precision using these primitives.

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
