# CubeIt

CubeIt is a lightweight quantum playground for **n-qubit** registers. Universal gate set and utilities for visualisation and testing.

## Features

- **N-Qubit Registers:** Create registers of any size with `quantumregister(n)`.
- **Universal Gate Set:** (`h()`, `s()`, `t()`, `cnot()`, …)
   (`qr.h(i)`, `qr.cnot(control, target)`).
- **Measurement:** `get_state()` prints amplitudes, `measure()`
  collapses and returns classical outcomes.
- **Utility Modules:** Measurement statistics, Bell-state builders, fidelity
  checks, and more under `cubeit.visualization`.

## Installation
Install the package
```
pip install cubeit
```

Or clone the repository and download the requirements

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from cubeit import quantumregister, get_state, measure

# 1) Create a 4-qubit register (|0000⟩)
qr = quantumregister(4)

# 2) Build a circuit
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

### Gates at a Glance

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


### `quantumregister`

Factory returning an instance of the internal `_QuantumRegister`. Methods:

- `apply_gate(matrix)` / `apply_single_qubit_gate(matrix, qubit)`
- `measure()` & `measure_qubit(qubit)`
- `get_state()` → returns a `QuantumState`
- `get_probabilities()` → `np.ndarray`
- Fluent helpers: `h`, `x`, `y`, `z`, `s`, `t`, `phase`, `rx`, `ry`, `rz`,
  `cnot`, `cz`, `cphase`, `swap`


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


## Requirements

- Python >= 3.8
- NumPy >= 1.20.0

## License

See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

