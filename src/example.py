"""
Comprehensive pytest test suite for CubeIt quantum software package.
"""

import pytest
import numpy as np
from cubeit import (
    QuantumState,
    quantumregister,
    get_state,
    measure,
    had, s, t, x, y, z,
    cnot, cnot_10, swap, cz,
)
from cubeit.visualization import fidelity

def create_bell_state(state_type: str = "phi_plus") -> quantumregister:
    """
    Create a Bell state (maximally entangled state).
    
    Args:
        state_type: Type of Bell state
            - "phi_plus":  (|00⟩ + |11⟩) / √2
            - "phi_minus": (|00⟩ - |11⟩) / √2
            - "psi_plus":  (|01⟩ + |10⟩) / √2
            - "psi_minus": (|01⟩ - |10⟩) / √2
    
    Returns:
        TwoQubitSystem in the specified Bell state
    """
    system = quantumregister(2)  # Bell states are for 2 qubits
    
    if state_type == "phi_plus":
        # |Φ⁺⟩ = (|00⟩ + |11⟩) / √2
        system.had(0)
        system.cnot(0, 1)
    elif state_type == "phi_minus":
        # |Φ⁻⟩ = (|00⟩ - |11⟩) / √2
        system.had(0)
        system.z(0)  # Apply Z before CNOT
        system.cnot(0, 1)
    elif state_type == "psi_plus":
        # |Ψ⁺⟩ = (|01⟩ + |10⟩) / √2
        system.had(0)
        system.x(1)  # Flip qubit 1
        system.cnot(0, 1)
    elif state_type == "psi_minus":
        # |Ψ⁻⟩ = (|01⟩ - |10⟩) / √2
        system.had(0)
        system.x(1)  # Flip qubit 1
        system.z(0)  # Apply Z before CNOT
        system.cnot(0, 1)
    else:
        raise ValueError(f"Unknown Bell state type: {state_type}")
    
    return system

class TestQuantumState:
    """Test cases for QuantumState class."""
    
    def test_initialization(self):
        """Test QuantumState initialization."""
        state = QuantumState(np.array([1.0, 0.0, 0.0, 0.0], dtype=complex))
        assert len(state.state) == 4
        assert np.allclose(np.abs(state.state) ** 2, [1.0, 0.0, 0.0, 0.0])
    
    def test_normalization(self):
        """Test that states are automatically normalized."""
        state = QuantumState(np.array([2.0, 0.0, 0.0, 0.0], dtype=complex))
        norm = np.linalg.norm(state.state)
        assert np.isclose(norm, 1.0)
    
    def test_zero_state_error(self):
        """Test that zero state raises error."""
        with pytest.raises(ValueError):
            QuantumState(np.array([0.0, 0.0, 0.0, 0.0], dtype=complex))
    
    def test_probabilities(self):
        """Test probability calculation."""
        state = QuantumState(np.array([1.0, 0.0, 0.0, 0.0], dtype=complex))
        probs = state.probabilities()
        assert np.allclose(probs, [1.0, 0.0, 0.0, 0.0])
        assert np.isclose(np.sum(probs), 1.0)
    
    def test_fidelity(self):
        """Test fidelity calculation."""
        state1 = QuantumState(np.array([1.0, 0.0, 0.0, 0.0], dtype=complex))
        state2 = QuantumState(np.array([1.0, 0.0, 0.0, 0.0], dtype=complex))
        assert np.isclose(state1.fidelity(state2), 1.0)
        
        state3 = QuantumState(np.array([0.0, 1.0, 0.0, 0.0], dtype=complex))
        assert np.isclose(state1.fidelity(state3), 0.0)


class TestQuantumRegisterBasics:
    """Tests for creating and validating quantum registers."""

    def test_initialization_default(self):
        """Default register should start in |00⟩."""
        register = quantumregister(2)
        state = register.get_state()
        assert np.allclose(state.state, [1.0, 0.0, 0.0, 0.0])

    def test_initialization_custom(self):
        """Custom initial states may be provided."""
        custom_state = QuantumState(np.array([0.0, 1.0, 0.0, 0.0], dtype=complex))
        register = quantumregister(2, custom_state)
        assert np.allclose(register.get_state().state, [0.0, 1.0, 0.0, 0.0])

    def test_initialization_wrong_size(self):
        """State dimension must match register size."""
        wrong_state = QuantumState(np.array([1.0, 0.0], dtype=complex))
        with pytest.raises(ValueError):
            quantumregister(2, wrong_state)


class TestUniversalGates:
    """Test cases for universal gate set (H, S, T, CNOT)."""
    
    def test_hadamard_qubit0(self):
        """Test Hadamard gate on qubit 0."""
        system = quantumregister(2)
        system.had(0)
        state = system.get_state().state
        expected = np.array([1/np.sqrt(2), 0.0, 1/np.sqrt(2), 0.0], dtype=complex)
        assert np.allclose(state, expected)
    
    def test_hadamard_qubit1(self):
        """Test Hadamard gate on qubit 1."""
        system = quantumregister(2)
        system.had(1)
        state = system.get_state().state
        expected = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0.0, 0.0], dtype=complex)
        assert np.allclose(state, expected)
    
    def test_phase_gate(self):
        """Test Phase (S) gate."""
        system = quantumregister(2)
        system.x(0)  # |10⟩
        system.s(0)  # Apply phase
        state = system.get_state().state
        expected = np.array([0.0, 0.0, 1.0, 0.0], dtype=complex) * 1j
        assert np.allclose(state, expected)
    
    def test_t_gate(self):
        """Test T gate."""
        system = quantumregister(2)
        system.x(0)  # |10⟩
        system.t(0)  # Apply T
        state = system.get_state().state
        expected = np.array([0.0, 0.0, 1.0, 0.0], dtype=complex) * np.exp(1j * np.pi / 4)
        assert np.allclose(state, expected)
    
    def test_cnot_creates_bell_state(self):
        """Test that H + CNOT creates Bell state."""
        system = quantumregister(2)
        system.had(0)
        system.cnot(0, 1)
        state = system.get_state().state
        expected = np.array([1/np.sqrt(2), 0.0, 0.0, 1/np.sqrt(2)], dtype=complex)
        assert np.allclose(state, expected)
    
    def test_cnot_10(self):
        """Test CNOT with control on qubit 1."""
        system = quantumregister(2)
        system.x(1)  # |01⟩
        system.apply_two_qubit_gate(cnot_10(), 0, 1)  # Control on qubit 1, target on qubit 0
        state = system.get_state().state
        expected = np.array([0.0, 0.0, 0.0, 1.0], dtype=complex)
        assert np.allclose(state, expected)


class TestPauliGates:
    """Test cases for Pauli gates (X, Y, Z)."""
    
    def test_pauli_x_qubit0(self):
        """Test Pauli-X (bit flip) on qubit 0."""
        system = quantumregister(2)
        system.x(0)
        state = system.get_state().state
        assert np.allclose(state, [0.0, 0.0, 1.0, 0.0])
    
    def test_pauli_x_qubit1(self):
        """Test Pauli-X (bit flip) on qubit 1."""
        system = quantumregister(2)
        system.x(1)
        state = system.get_state().state
        assert np.allclose(state, [0.0, 1.0, 0.0, 0.0])
    
    def test_pauli_x_twice(self):
        """Test that X applied twice returns to original state."""
        system = quantumregister(2)
        system.x(0)
        system.x(0)
        state = system.get_state().state
        assert np.allclose(state, [1.0, 0.0, 0.0, 0.0])
    
    def test_pauli_y(self):
        """Test Pauli-Y gate."""
        system = quantumregister(2)
        system.y(0)
        state = system.get_state().state
        expected = np.array([0.0, 0.0, 1j, 0.0], dtype=complex)
        assert np.allclose(state, expected)
    
    def test_pauli_z(self):
        """Test Pauli-Z (phase flip) gate."""
        system = quantumregister(2)
        system.had(0)  # Create superposition
        system.z(0)  # Apply phase flip
        state = system.get_state().state
        expected = np.array([1/np.sqrt(2), 0.0, -1/np.sqrt(2), 0.0], dtype=complex)
        assert np.allclose(state, expected)


class TestParameterizedGates:
    """Test cases for parameterized gates."""
    
    def test_phase_gate(self):
        """Test parameterized Phase gate."""
        system = quantumregister(2)
        system.x(0)  # |10⟩
        system.phase(0, np.pi/2)
        state = system.get_state().state
        expected = np.array([0.0, 0.0, 1.0, 0.0], dtype=complex) * 1j
        assert np.allclose(state, expected)
    
    def test_rotation_x(self):
        """Test RotationX gate."""
        system = quantumregister(2)
        system.rx(0, np.pi)
        state = system.get_state().state
        # Rx(π) on |0⟩ gives -i|1⟩, so Rx(π)|00⟩ = -i|10⟩
        # Check that we have |10⟩ up to a global phase
        assert np.isclose(abs(state[2]), 1.0)  # |10⟩ component should have magnitude 1
        assert np.allclose(state[[0, 1, 3]], [0.0, 0.0, 0.0])  # Other components should be 0
    
    def test_rotation_y(self):
        """Test RotationY gate."""
        system = quantumregister(2)
        system.ry(0, np.pi)
        state = system.get_state().state
        # Ry(π) should flip the qubit
        assert np.allclose(state, [0.0, 0.0, 1.0, 0.0])
    
    def test_rotation_z(self):
        """Test RotationZ gate."""
        system = quantumregister(2)
        system.had(0)  # Create superposition (|0⟩ + |1⟩)/√2
        system.rz(0, np.pi)
        state = system.get_state().state
        # Rz(π) on (|0⟩ + |1⟩)/√2 gives (-i|0⟩ + i|1⟩)/√2 = i(|1⟩ - |0⟩)/√2
        # Check that we have the correct relative phase (|0⟩ and |1⟩ components have opposite phases)
        assert np.isclose(abs(state[0]), 1/np.sqrt(2))  # |00⟩ component
        assert np.isclose(abs(state[2]), 1/np.sqrt(2))  # |10⟩ component
        assert np.allclose(state[[1, 3]], [0.0, 0.0])  # Other components should be 0
        # Check that the phases are opposite (up to global phase)
        phase_diff = np.angle(state[2]) - np.angle(state[0])
        assert np.isclose(phase_diff % (2*np.pi), np.pi % (2*np.pi), atol=1e-6) or \
               np.isclose(phase_diff % (2*np.pi), -np.pi % (2*np.pi), atol=1e-6)


class TestTwoQubitGates:
    """Test cases for two-qubit gates."""
    
    def test_swap_gate(self):
        """Test SWAP gate."""
        system = quantumregister(2)
        system.x(0)  # |10⟩
        system.swap(0, 1)  # Should become |01⟩
        state = system.get_state().state
        assert np.allclose(state, [0.0, 1.0, 0.0, 0.0])
    
    def test_cz_gate(self):
        """Test Controlled-Z gate."""
        system = quantumregister(2)
        system.had(0)
        system.had(1)
        system.cz(0, 1)
        # CZ should add a phase to |11⟩
        probs = system.get_probabilities()
        assert np.isclose(np.sum(probs), 1.0)
    
    def test_cphase_gate(self):
        """Test Controlled-Phase gate."""
        system = quantumregister(2)
        system.x(0)
        system.x(1)  # |11⟩
        system.cphase(0, 1, np.pi/2)
        state = system.get_state().state
        expected = np.array([0.0, 0.0, 0.0, 1.0], dtype=complex) * np.exp(1j * np.pi / 2)
        assert np.allclose(state, expected)


class TestMeasurements:
    """Test cases for measurement operations."""
    
    def test_measurement_deterministic(self):
        """Test measurement of deterministic state."""
        system = quantumregister(2)  # |00⟩
        result = system.measure()
        assert result == (0, 0)
    
    def test_measurement_probabilities(self):
        """Test that probabilities sum to 1."""
        system = quantumregister(2)
        system.had(0)
        probs = system.get_probabilities()
        assert np.isclose(np.sum(probs), 1.0)
        assert len(probs) == 4
    
    def test_measurement_statistics(self):
        """Test measurement statistics over many samples."""
        system = quantumregister(2)
        system.had(0)
        system.cnot(0, 1)  # Bell state
        
        # Measure many times
        results = [system.measure() for _ in range(1000)]
        outcomes = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
        for result in results:
            outcomes[result] += 1
        
        # Bell state should only give |00⟩ or |11⟩
        assert outcomes[(0, 1)] == 0
        assert outcomes[(1, 0)] == 0
        assert outcomes[(0, 0)] + outcomes[(1, 1)] == 1000
    
    def test_partial_measurement_qubit0(self):
        """Test partial measurement of qubit 0."""
        system = quantumregister(2)
        system.had(0)
        result = system.measure_qubit(0)
        assert result in [0, 1]
        # After measurement, state should be collapsed
        probs = system.get_probabilities()
        assert np.isclose(np.sum(probs), 1.0)
    
    def test_partial_measurement_qubit1(self):
        """Test partial measurement of qubit 1."""
        system = quantumregister(2)
        system.had(1)
        result = system.measure_qubit(1)
        assert result in [0, 1]
        probs = system.get_probabilities()
        assert np.isclose(np.sum(probs), 1.0)


class TestBellStates:
    """Test cases for Bell states."""
    
    def test_bell_phi_plus(self):
        """Test |Φ⁺⟩ Bell state."""
        bell = create_bell_state("phi_plus")
        state = bell.get_state().state
        expected = np.array([1/np.sqrt(2), 0.0, 0.0, 1/np.sqrt(2)], dtype=complex)
        assert np.allclose(state, expected)
    
    def test_bell_phi_minus(self):
        """Test |Φ⁻⟩ Bell state."""
        bell = create_bell_state("phi_minus")
        state = bell.get_state().state
        expected = np.array([1/np.sqrt(2), 0.0, 0.0, -1/np.sqrt(2)], dtype=complex)
        assert np.allclose(state, expected)
    
    def test_bell_psi_plus(self):
        """Test |Ψ⁺⟩ Bell state."""
        bell = create_bell_state("psi_plus")
        state = bell.get_state().state
        expected = np.array([0.0, 1/np.sqrt(2), 1/np.sqrt(2), 0.0], dtype=complex)
        assert np.allclose(state, expected)
    
    def test_bell_psi_minus(self):
        """Test |Ψ⁻⟩ Bell state."""
        bell = create_bell_state("psi_minus")
        state = bell.get_state().state
        expected = np.array([0.0, 1/np.sqrt(2), -1/np.sqrt(2), 0.0], dtype=complex)
        assert np.allclose(state, expected)
    
    def test_bell_state_entanglement(self):
        """Test that Bell states are maximally entangled."""
        bell = create_bell_state("phi_plus")
        probs = bell.get_probabilities()
        # Bell state should have equal probability for |00⟩ and |11⟩
        assert np.isclose(probs[0], 0.5)
        assert np.isclose(probs[3], 0.5)
        assert np.isclose(probs[1], 0.0)
        assert np.isclose(probs[2], 0.0)


class TestGateCompositions:
    """Test cases for gate compositions and circuits."""
    
    def test_h_x_h(self):
        """Test H X H = Z (up to global phase)."""
        system1 = quantumregister(2)
        system1.had(0)
        system1.x(0)
        system1.had(0)
        
        system2 = quantumregister(2)
        system2.z(0)
        
        # States should be equivalent (up to global phase)
        state1 = system1.get_state().state
        state2 = system2.get_state().state
        # For |00⟩, both should give |00⟩
        assert np.allclose(np.abs(state1), np.abs(state2))
    
    def test_cnot_self_inverse(self):
        """Test that CNOT is its own inverse."""
        system = quantumregister(2)
        system.had(0)
        system.cnot(0, 1)
        original_state = system.get_state().state.copy()
        system.cnot(0, 1)  # Apply again
        system.had(0)
        final_state = system.get_state().state
        assert np.allclose(final_state, [1.0, 0.0, 0.0, 0.0])
    
    def test_swap_via_cnot(self):
        """Test that SWAP can be decomposed into CNOTs."""
        system1 = quantumregister(2)
        system1.x(0)  # |10⟩
        system1.swap(0, 1)
        
        system2 = quantumregister(2)
        system2.x(0)  # |10⟩
        system2.cnot(0, 1)
        system2.apply_two_qubit_gate(cnot_10(), 0, 1)
        system2.cnot(0, 1)
        
        state1 = system1.get_state().state
        state2 = system2.get_state().state
        assert np.allclose(state1, state2)


class TestUnitarity:
    """Test cases for gate unitarity."""
    
    def test_hadamard_unitary(self):
        """Test that Hadamard gate is unitary."""
        gate = had()
        h_dagger = gate.conj().T
        product = gate @ h_dagger
        assert np.allclose(product, np.eye(2))
    
    def test_cnot_unitary(self):
        """Test that CNOT gate is unitary."""
        gate = cnot()
        cnot_dagger = gate.conj().T
        product = gate @ cnot_dagger
        assert np.allclose(product, np.eye(4))
    
    def test_all_single_qubit_gates_unitary(self):
        """Test that all single-qubit gates are unitary."""
        gates = [had(), s(), t(), x(), y(), z()]
        for gate in gates:
            gate_dagger = gate.conj().T
            product = gate @ gate_dagger
            assert np.allclose(product, np.eye(2), atol=1e-10)
    
    def test_all_two_qubit_gates_unitary(self):
        """Test that all two-qubit gates are unitary."""
        gates = [cnot(), cnot_10(), swap(), cz()]
        for gate in gates:
            gate_dagger = gate.conj().T
            product = gate @ gate_dagger
            assert np.allclose(product, np.eye(4), atol=1e-10)


class TestErrorHandling:
    """Test cases for error handling."""
    
    def test_invalid_qubit_index(self):
        """Test that invalid qubit index raises error."""
        system = quantumregister(2)
        with pytest.raises(ValueError):
            system.had(2)
    
    def test_wrong_gate_size(self):
        """Test that wrong gate size raises error."""
        system = quantumregister(2)
        wrong_gate = np.eye(3)  # 3x3 instead of 2x2
        with pytest.raises(ValueError):
            system.apply_single_qubit_gate(wrong_gate, 0)
    
    def test_wrong_two_qubit_gate_size(self):
        """Test that wrong two-qubit gate size raises error."""
        system = quantumregister(2)
        wrong_gate = np.eye(2)  # 2x2 instead of 4x4
        with pytest.raises(ValueError):
            system.apply_gate(wrong_gate)
    
    def test_invalid_bell_state_type(self):
        """Test that invalid Bell state type raises error."""
        with pytest.raises(ValueError):
            create_bell_state("invalid")


class TestFidelity:
    """Test cases for fidelity calculations."""
    
    def test_fidelity_identical_states(self):
        """Test fidelity of identical states."""
        system1 = quantumregister(2)
        system2 = quantumregister(2)
        assert np.isclose(fidelity(system1, system2), 1.0)
    
    def test_fidelity_orthogonal_states(self):
        """Test fidelity of orthogonal states."""
        system1 = quantumregister(2)  # |00⟩
        system2 = quantumregister(2)
        system2.x(0)
        system2.x(1)  # |11⟩
        assert np.isclose(fidelity(system1, system2), 0.0)
    
    def test_fidelity_bell_states(self):
        """Test fidelity between Bell states."""
        bell1 = create_bell_state("phi_plus")
        bell2 = create_bell_state("phi_plus")
        assert np.isclose(fidelity(bell1, bell2), 1.0)


class TestNQubitRegister:
    """Test cases for n-qubit QuantumRegister."""
    
    def test_single_qubit_register(self):
        """Test 1-qubit register."""
        reg = quantumregister(1)
        assert reg.num_qubits == 1
        assert reg.dim == 2
        state = reg.get_state()
        assert np.allclose(state.state, [1.0, 0.0])
        
        # Apply Hadamard
        reg.had(0)
        state = reg.get_state().state
        expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)], dtype=complex)
        assert np.allclose(state, expected)
    
    def test_three_qubit_register(self):
        """Test 3-qubit register."""
        reg = quantumregister(3)
        assert reg.num_qubits == 3
        assert reg.dim == 8
        state = reg.get_state()
        assert len(state.state) == 8
        assert np.allclose(state.state[0], 1.0)
        assert np.allclose(state.state[1:], 0.0)
    
    def test_four_qubit_register(self):
        """Test 4-qubit register."""
        reg = quantumregister(4)
        assert reg.num_qubits == 4
        assert reg.dim == 16
        state = reg.get_state()
        assert len(state.state) == 16
    
    def test_single_qubit_gate_on_n_qubits(self):
        """Test applying single-qubit gates on n-qubit register."""
        reg = quantumregister(3)
        
        # Apply Had to qubit 0
        reg.had(0)
        state = reg.get_state().state
        # Should be (|000⟩ + |100⟩) / √2 (qubit indices are ordered left-to-right)
        expected_indices = [0, 4]  # |000⟩ and |100⟩
        assert np.isclose(abs(state[expected_indices[0]]), 1/np.sqrt(2))
        assert np.isclose(abs(state[expected_indices[1]]), 1/np.sqrt(2))
        mask = np.ones_like(state, dtype=bool)
        mask[expected_indices] = False
        assert np.allclose(state[mask], 0.0)
        
        # Apply H to qubit 1
        reg = quantumregister(3)
        reg.had(1)
        state = reg.get_state().state
        # Should be (|000⟩ + |010⟩) / √2
        expected_indices = [0, 2]  # |000⟩ and |010⟩
        assert np.isclose(abs(state[expected_indices[0]]), 1/np.sqrt(2))
        assert np.isclose(abs(state[expected_indices[1]]), 1/np.sqrt(2))
        mask = np.ones_like(state, dtype=bool)
        mask[expected_indices] = False
        assert np.allclose(state[mask], 0.0)
    
    def test_two_qubit_gate_on_n_qubits(self):
        """Test applying two-qubit gates on n-qubit register."""
        reg = quantumregister(3)
        
        # Apply H to qubit 0, then CNOT between qubit 0 and 1
        reg.had(0)
        reg.cnot(0, 1)
        state = reg.get_state().state
        
        # Should be (|000⟩ + |110⟩) / √2
        expected_indices = [0, 6]  # |000⟩ and |110⟩
        assert np.isclose(abs(state[expected_indices[0]]), 1/np.sqrt(2))
        assert np.isclose(abs(state[expected_indices[1]]), 1/np.sqrt(2))
        mask = np.ones_like(state, dtype=bool)
        mask[expected_indices] = False
        assert np.allclose(state[mask], 0.0)
    
    def test_cnot_on_different_qubits(self):
        """Test CNOT on different qubit pairs in n-qubit register."""
        reg = quantumregister(4)
        
        # Apply H to qubit 0, then CNOT between qubit 0 and 2
        reg.had(0)
        reg.cnot(0, 2)
        state = reg.get_state().state
        
        # Should be (|0000⟩ + |1010⟩) / √2
        expected_indices = [0, 10]  # |0000⟩ and |1010⟩
        assert np.isclose(abs(state[expected_indices[0]]), 1/np.sqrt(2))
        assert np.isclose(abs(state[expected_indices[1]]), 1/np.sqrt(2))
    
    def test_measurement_n_qubits(self):
        """Test measurement on n-qubit register."""
        reg = quantumregister(3)
        result = reg.measure()
        assert len(result) == 3
        assert all(bit in [0, 1] for bit in result)
        assert result == (0, 0, 0)  # Should be |000⟩
    
    def test_partial_measurement_n_qubits(self):
        """Test partial measurement on n-qubit register."""
        reg = quantumregister(3)
        reg.had(0)
        
        # Measure qubit 0
        result = reg.measure_qubit(0)
        assert result in [0, 1]
        
        # State should be collapsed
        probs = reg.get_probabilities()
        assert np.isclose(np.sum(probs), 1.0)
    
    def test_ghz_state_three_qubits(self):
        """Test creating GHZ state on 3 qubits."""
        reg = quantumregister(3)
        reg.had(0)
        reg.cnot(0, 1)
        reg.cnot(1, 2)
        
        state = reg.get_state().state
        # GHZ state: (|000⟩ + |111⟩) / √2
        assert np.isclose(abs(state[0]), 1/np.sqrt(2))
        assert np.isclose(abs(state[7]), 1/np.sqrt(2))
        assert np.allclose(state[1:7], 0.0)
    
    def test_ghz_state_four_qubits(self):
        """Test creating GHZ state on 4 qubits."""
        reg = quantumregister(4)
        reg.had(0)
        reg.cnot(0, 1)
        reg.cnot(1, 2)
        reg.cnot(2, 3)
        
        state = reg.get_state().state
        # GHZ state: (|0000⟩ + |1111⟩) / √2
        assert np.isclose(abs(state[0]), 1/np.sqrt(2))
        assert np.isclose(abs(state[15]), 1/np.sqrt(2))
        assert np.allclose(state[1:15], 0.0)
    
    def test_swap_on_n_qubits(self):
        """Test SWAP gate on n-qubit register."""
        reg = quantumregister(3)
        reg.x(0)  # |001⟩
        reg.swap(0, 1)  # Should become |010⟩
        
        state = reg.get_state().state
        assert np.isclose(abs(state[2]), 1.0)  # |010⟩
        assert np.allclose(state[[0, 1, 3, 4, 5, 6, 7]], 0.0)
    
    def test_state_string_representation(self):
        """Test state string representation for n qubits."""
        reg = quantumregister(3)
        state_str = reg.get_state().to_string()
        assert "|000⟩" in state_str or "1.000|000⟩" in state_str
        
        reg.had(0)
        state_str = reg.get_state().to_string()
        assert "|000⟩" in state_str
        assert "|100⟩" in state_str
    
    def test_invalid_qubit_index(self):
        """Test that invalid qubit index raises error."""
        reg = quantumregister(3)
        with pytest.raises(ValueError):
            reg.had(3)  # Only 0, 1, 2 are valid
        
        with pytest.raises(ValueError):
            reg.cnot(0, 3)
    
    def test_quantum_state_n_qubits(self):
        """Test QuantumState with n qubits."""
        # 1 qubit
        state1 = QuantumState(np.array([1.0, 0.0], dtype=complex))
        assert state1.num_qubits == 1
        
        # 3 qubits
        state3 = QuantumState(np.array([1.0] + [0.0] * 7, dtype=complex))
        assert state3.num_qubits == 3
        assert len(state3.state) == 8
        
        # 4 qubits
        state4 = QuantumState(np.array([1.0] + [0.0] * 15, dtype=complex))
        assert state4.num_qubits == 4
        assert len(state4.state) == 16
    
    def test_gate_unitarity_n_qubits(self):
        """Test that gates remain unitary on n-qubit systems."""
        reg = quantumregister(3)
        
        # Apply various gates
        reg.had(0)
        reg.s(1)
        reg.cnot(0, 2)
        
        # State should still be normalized
        probs = reg.get_probabilities()
        assert np.isclose(np.sum(probs), 1.0)
    
    def test_measurement_statistics_n_qubits(self):
        """Test measurement statistics on n-qubit register."""
        reg = quantumregister(3)
        reg.had(0)
        reg.cnot(0, 1)
        
        # Measure many times
        results = [reg.measure() for _ in range(100)]
        assert all(len(r) == 3 for r in results)
        assert all(bit in [0, 1] for r in results for bit in r)
        
        # Should only get |000⟩ or |011⟩
        unique_results = set(results)
        assert len(unique_results) <= 2
        assert (0, 0, 0) in unique_results or (0, 1, 1) in unique_results


class TestUserFriendlyAPI:
    """Tests for the convenience helpers exposed on QuantumRegister."""

    def test_factory_function(self):
        reg = quantumregister(3)
        assert reg.__class__.__name__ == "_QuantumRegister"
        assert reg.num_qubits == 3

    def test_single_qubit_helper(self):
        reg = quantumregister(2)
        reg.had(0)
        state = reg.get_state().state
        # (|00⟩ + |10⟩)/√2 → indices 0 and 2
        assert np.isclose(abs(state[0]), 1/np.sqrt(2))
        assert np.isclose(abs(state[2]), 1/np.sqrt(2))
        mask = np.ones_like(state, dtype=bool)
        mask[[0, 2]] = False
        assert np.allclose(state[mask], 0.0)

    def test_two_qubit_helper(self):
        reg = quantumregister(2)
        reg.had(0).cnot(0, 1)
        state = reg.get_state().state
        expected = np.array([1/np.sqrt(2), 0.0, 0.0, 1/np.sqrt(2)], dtype=complex)
        assert np.allclose(state, expected)

    def test_chained_operations(self):
        reg = quantumregister(3)
        reg.had(0).x(2).cnot(0, 1).swap(1, 2)
        # Basic sanity checks on normalization and amplitudes
        probs = reg.get_probabilities()
        assert np.isclose(np.sum(probs), 1.0)

    def test_twoqubitsystem_helpers(self):
        system = quantumregister(2)
        system.had(0).cnot(0, 1)
        state = system.get_state().state
        expected = np.array([1/np.sqrt(2), 0.0, 0.0, 1/np.sqrt(2)], dtype=complex)
        assert np.allclose(state, expected)

    def test_module_get_state_function(self, capsys):
        reg = quantumregister(2).had(0)
        description = get_state(reg)
        captured = capsys.readouterr()
        assert "|00" in captured.out
        assert isinstance(description, str)

    def test_module_get_state_array(self):
        reg = quantumregister(1)
        array = get_state(reg, print_state=False, as_array=True)
        assert array.shape == (2,)
        assert np.allclose(array, [1.0, 0.0])

    def test_module_measure_single_qubit(self, capsys):
        reg = quantumregister(1).had(0)
        result = measure(reg)
        captured = capsys.readouterr()
        assert result in (0, 1)
        assert captured.out.strip() in {"0", "1"}

    def test_module_measure_multi_qubit(self):
        reg = quantumregister(2).had(0).cnot(0, 1)
        result = measure(reg, print_result=False)
        assert result in ((0, 0), (1, 1))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
