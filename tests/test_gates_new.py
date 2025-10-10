"""Tests for new single-qubit gates: Y, Sdag, T, Tdag."""

import pytest
import numpy as np


class TestPauliYGate:
    """Test Pauli-Y gate."""

    def test_y_matrix_definition(self):
        """Test Pauli-Y gate matrix definition."""
        from feynman_path.core.gates import GATE_MATRICES

        Y = GATE_MATRICES['Y']
        expected = np.array([[0, -1j], [1j, 0]])

        assert np.allclose(Y, expected)

    def test_y_matrix_shape(self):
        """Test Y matrix is 2x2."""
        from feynman_path.core.gates import GATE_MATRICES

        Y = GATE_MATRICES['Y']
        assert Y.shape == (2, 2)

    def test_y_is_unitary(self):
        """Test that Y is unitary: Y†Y = I."""
        from feynman_path.core.gates import GATE_MATRICES

        Y = GATE_MATRICES['Y']
        assert np.allclose(Y @ Y.conj().T, np.eye(2))

    def test_y_squared_is_identity(self):
        """Test that Y² = I (not -I as in some conventions)."""
        from feynman_path.core.gates import GATE_MATRICES

        Y = GATE_MATRICES['Y']
        # For Y = [[0, -i], [i, 0]], Y² = I (not -I)
        Y_squared = Y @ Y
        assert np.allclose(Y_squared, np.eye(2))

    def test_y_hermitian(self):
        """Test that Y is Hermitian: Y† = Y."""
        from feynman_path.core.gates import GATE_MATRICES

        Y = GATE_MATRICES['Y']
        assert np.allclose(Y, Y.conj().T)

    def test_y_apply_to_zero_state(self):
        """Test Y|0⟩ = i|1⟩."""
        from feynman_path.core.gates import Gate, GATE_MATRICES

        gate = Gate('Y', GATE_MATRICES['Y'], qubits=(0,), n_qubits=1)
        transitions = gate.apply_to_state('0')

        assert len(transitions) == 1
        assert '1' in transitions
        assert abs(transitions['1'] - 1j) < 1e-10

    def test_y_apply_to_one_state(self):
        """Test Y|1⟩ = -i|0⟩."""
        from feynman_path.core.gates import Gate, GATE_MATRICES

        gate = Gate('Y', GATE_MATRICES['Y'], qubits=(0,), n_qubits=1)
        transitions = gate.apply_to_state('1')

        assert len(transitions) == 1
        assert '0' in transitions
        assert abs(transitions['0'] - (-1j)) < 1e-10

    def test_y_on_specific_qubit_in_three_qubit_system(self):
        """Test Y gate on middle qubit (q1) in 3-qubit system."""
        from feynman_path.core.gates import Gate, GATE_MATRICES

        gate = Gate('Y', GATE_MATRICES['Y'], qubits=(1,), n_qubits=3)

        # Test |010⟩ (q0=0, q1=1, q2=0) → Y on q1
        # Y|1⟩ = -i|0⟩, so |010⟩ → -i|000⟩
        transitions = gate.apply_to_state('010')

        assert '000' in transitions
        assert abs(transitions['000'] - (-1j)) < 1e-10

    def test_y_commutation_with_z(self):
        """Test that YZ = -ZY (anti-commutation)."""
        from feynman_path.core.gates import GATE_MATRICES

        Y = GATE_MATRICES['Y']
        Z = GATE_MATRICES['Z']

        YZ = Y @ Z
        ZY = Z @ Y

        assert np.allclose(YZ, -ZY)

    def test_y_gate_from_registry(self):
        """Test creating Y gate from registry."""
        from feynman_path.core.gates import GATE_REGISTRY

        gate = GATE_REGISTRY['y'](qubit=0, n_qubits=2)

        assert gate.name == 'Y'
        assert gate.qubits == (0,)
        assert gate.n_qubits == 2

    def test_parse_y_gate_spec(self):
        """Test parsing Y gate specification."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('y0', n_qubits=2)

        assert gate.name == 'Y'
        assert gate.qubits == (0,)

    def test_y_in_multi_qubit_circuit(self):
        """Test Y gate in a 2-qubit circuit."""
        from feynman_path.core.gates import Gate, GATE_MATRICES

        # Y on qubit 1 in 2-qubit system
        gate = Gate('Y', GATE_MATRICES['Y'], qubits=(1,), n_qubits=2)

        # Test |01⟩ (q0=0, q1=1) → -i|00⟩
        transitions = gate.apply_to_state('01')
        assert transitions == {'00': -1j}

        # Test |11⟩ (q0=1, q1=1) → -i|10⟩
        transitions = gate.apply_to_state('11')
        assert abs(transitions['10'] - (-1j)) < 1e-10


class TestSdagGate:
    """Test S† (S-dagger, adjoint of S) gate."""

    def test_sdag_matrix_definition(self):
        """Test S† gate matrix definition."""
        from feynman_path.core.gates import GATE_MATRICES

        Sdag = GATE_MATRICES['Sdag']
        expected = np.array([[1, 0], [0, -1j]])

        assert np.allclose(Sdag, expected)

    def test_sdag_matrix_shape(self):
        """Test S† matrix is 2x2."""
        from feynman_path.core.gates import GATE_MATRICES

        Sdag = GATE_MATRICES['Sdag']
        assert Sdag.shape == (2, 2)

    def test_sdag_is_unitary(self):
        """Test that S† is unitary."""
        from feynman_path.core.gates import GATE_MATRICES

        Sdag = GATE_MATRICES['Sdag']
        assert np.allclose(Sdag @ Sdag.conj().T, np.eye(2))

    def test_sdag_is_adjoint_of_s(self):
        """Test that S† = S*."""
        from feynman_path.core.gates import GATE_MATRICES

        S = GATE_MATRICES['S']
        Sdag = GATE_MATRICES['Sdag']

        assert np.allclose(Sdag, S.conj().T)

    def test_s_times_sdag_is_identity(self):
        """Test that S·S† = I."""
        from feynman_path.core.gates import GATE_MATRICES

        S = GATE_MATRICES['S']
        Sdag = GATE_MATRICES['Sdag']

        assert np.allclose(S @ Sdag, np.eye(2))
        assert np.allclose(Sdag @ S, np.eye(2))

    def test_sdag_apply_to_zero_state(self):
        """Test S†|0⟩ = |0⟩."""
        from feynman_path.core.gates import Gate, GATE_MATRICES

        gate = Gate('Sdag', GATE_MATRICES['Sdag'], qubits=(0,), n_qubits=1)
        transitions = gate.apply_to_state('0')

        assert transitions == {'0': 1.0}

    def test_sdag_apply_to_one_state(self):
        """Test S†|1⟩ = -i|1⟩."""
        from feynman_path.core.gates import Gate, GATE_MATRICES

        gate = Gate('Sdag', GATE_MATRICES['Sdag'], qubits=(0,), n_qubits=1)
        transitions = gate.apply_to_state('1')

        assert len(transitions) == 1
        assert '1' in transitions
        assert abs(transitions['1'] - (-1j)) < 1e-10

    def test_sdag_on_specific_qubit(self):
        """Test S† on specific qubit in multi-qubit system."""
        from feynman_path.core.gates import Gate, GATE_MATRICES

        gate = Gate('Sdag', GATE_MATRICES['Sdag'], qubits=(1,), n_qubits=3)

        # Test |010⟩ → -i|010⟩ (q1=1 gets phase -i)
        transitions = gate.apply_to_state('010')
        assert abs(transitions['010'] - (-1j)) < 1e-10

    def test_parse_sdag_gate_spec(self):
        """Test parsing S† gate specification."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('sdag0', n_qubits=2)

        assert gate.name == 'Sdag'
        assert gate.qubits == (0,)

    def test_sdag_from_registry(self):
        """Test creating S† gate from registry."""
        from feynman_path.core.gates import GATE_REGISTRY

        gate = GATE_REGISTRY['sdag'](qubit=1, n_qubits=2)

        assert gate.name == 'Sdag'
        assert gate.qubits == (1,)


class TestTGate:
    """Test T gate (π/8 gate)."""

    def test_t_matrix_definition(self):
        """Test T gate matrix definition."""
        from feynman_path.core.gates import GATE_MATRICES

        T = GATE_MATRICES['T']
        expected = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])

        assert np.allclose(T, expected)

    def test_t_matrix_shape(self):
        """Test T matrix is 2x2."""
        from feynman_path.core.gates import GATE_MATRICES

        T = GATE_MATRICES['T']
        assert T.shape == (2, 2)

    def test_t_is_unitary(self):
        """Test that T is unitary."""
        from feynman_path.core.gates import GATE_MATRICES

        T = GATE_MATRICES['T']
        assert np.allclose(T @ T.conj().T, np.eye(2))

    def test_t_squared_is_s(self):
        """Test that T² = S."""
        from feynman_path.core.gates import GATE_MATRICES

        T = GATE_MATRICES['T']
        S = GATE_MATRICES['S']

        T_squared = T @ T
        assert np.allclose(T_squared, S)

    def test_t_to_fourth_power_is_z(self):
        """Test that T⁴ = Z."""
        from feynman_path.core.gates import GATE_MATRICES

        T = GATE_MATRICES['T']
        Z = GATE_MATRICES['Z']

        T_fourth = T @ T @ T @ T
        assert np.allclose(T_fourth, Z)

    def test_t_apply_to_zero_state(self):
        """Test T|0⟩ = |0⟩."""
        from feynman_path.core.gates import Gate, GATE_MATRICES

        gate = Gate('T', GATE_MATRICES['T'], qubits=(0,), n_qubits=1)
        transitions = gate.apply_to_state('0')

        assert transitions == {'0': 1.0}

    def test_t_apply_to_one_state(self):
        """Test T|1⟩ = e^(iπ/4)|1⟩."""
        from feynman_path.core.gates import Gate, GATE_MATRICES

        gate = Gate('T', GATE_MATRICES['T'], qubits=(0,), n_qubits=1)
        transitions = gate.apply_to_state('1')

        expected_phase = np.exp(1j * np.pi / 4)
        assert len(transitions) == 1
        assert '1' in transitions
        assert abs(transitions['1'] - expected_phase) < 1e-10

    def test_t_on_specific_qubit(self):
        """Test T on specific qubit in multi-qubit system."""
        from feynman_path.core.gates import Gate, GATE_MATRICES

        gate = Gate('T', GATE_MATRICES['T'], qubits=(0,), n_qubits=2)

        # Test |10⟩ (q0=1, q1=0) → e^(iπ/4)|10⟩
        transitions = gate.apply_to_state('10')
        expected_phase = np.exp(1j * np.pi / 4)
        assert abs(transitions['10'] - expected_phase) < 1e-10

    def test_parse_t_gate_spec(self):
        """Test parsing T gate specification."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('t0', n_qubits=2)

        assert gate.name == 'T'
        assert gate.qubits == (0,)

    def test_t_from_registry(self):
        """Test creating T gate from registry."""
        from feynman_path.core.gates import GATE_REGISTRY

        gate = GATE_REGISTRY['t'](qubit=0, n_qubits=3)

        assert gate.name == 'T'
        assert gate.qubits == (0,)


class TestTdagGate:
    """Test T† (T-dagger, adjoint of T) gate."""

    def test_tdag_matrix_definition(self):
        """Test T† gate matrix definition."""
        from feynman_path.core.gates import GATE_MATRICES

        Tdag = GATE_MATRICES['Tdag']
        expected = np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]])

        assert np.allclose(Tdag, expected)

    def test_tdag_matrix_shape(self):
        """Test T† matrix is 2x2."""
        from feynman_path.core.gates import GATE_MATRICES

        Tdag = GATE_MATRICES['Tdag']
        assert Tdag.shape == (2, 2)

    def test_tdag_is_unitary(self):
        """Test that T† is unitary."""
        from feynman_path.core.gates import GATE_MATRICES

        Tdag = GATE_MATRICES['Tdag']
        assert np.allclose(Tdag @ Tdag.conj().T, np.eye(2))

    def test_tdag_is_adjoint_of_t(self):
        """Test that T† = T*."""
        from feynman_path.core.gates import GATE_MATRICES

        T = GATE_MATRICES['T']
        Tdag = GATE_MATRICES['Tdag']

        assert np.allclose(Tdag, T.conj().T)

    def test_t_times_tdag_is_identity(self):
        """Test that T·T† = I."""
        from feynman_path.core.gates import GATE_MATRICES

        T = GATE_MATRICES['T']
        Tdag = GATE_MATRICES['Tdag']

        assert np.allclose(T @ Tdag, np.eye(2))
        assert np.allclose(Tdag @ T, np.eye(2))

    def test_tdag_squared_is_sdag(self):
        """Test that (T†)² = S†."""
        from feynman_path.core.gates import GATE_MATRICES

        Tdag = GATE_MATRICES['Tdag']
        Sdag = GATE_MATRICES['Sdag']

        Tdag_squared = Tdag @ Tdag
        assert np.allclose(Tdag_squared, Sdag)

    def test_tdag_apply_to_zero_state(self):
        """Test T†|0⟩ = |0⟩."""
        from feynman_path.core.gates import Gate, GATE_MATRICES

        gate = Gate('Tdag', GATE_MATRICES['Tdag'], qubits=(0,), n_qubits=1)
        transitions = gate.apply_to_state('0')

        assert transitions == {'0': 1.0}

    def test_tdag_apply_to_one_state(self):
        """Test T†|1⟩ = e^(-iπ/4)|1⟩."""
        from feynman_path.core.gates import Gate, GATE_MATRICES

        gate = Gate('Tdag', GATE_MATRICES['Tdag'], qubits=(0,), n_qubits=1)
        transitions = gate.apply_to_state('1')

        expected_phase = np.exp(-1j * np.pi / 4)
        assert len(transitions) == 1
        assert '1' in transitions
        assert abs(transitions['1'] - expected_phase) < 1e-10

    def test_tdag_on_specific_qubit(self):
        """Test T† on specific qubit in multi-qubit system."""
        from feynman_path.core.gates import Gate, GATE_MATRICES

        gate = Gate('Tdag', GATE_MATRICES['Tdag'], qubits=(2,), n_qubits=3)

        # Test |001⟩ (q2=1) → e^(-iπ/4)|001⟩
        transitions = gate.apply_to_state('001')
        expected_phase = np.exp(-1j * np.pi / 4)
        assert abs(transitions['001'] - expected_phase) < 1e-10

    def test_parse_tdag_gate_spec(self):
        """Test parsing T† gate specification."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('tdag0', n_qubits=2)

        assert gate.name == 'Tdag'
        assert gate.qubits == (0,)

    def test_tdag_from_registry(self):
        """Test creating T† gate from registry."""
        from feynman_path.core.gates import GATE_REGISTRY

        gate = GATE_REGISTRY['tdag'](qubit=1, n_qubits=2)

        assert gate.name == 'Tdag'
        assert gate.qubits == (1,)


class TestNewGatesIntegration:
    """Integration tests for new gates."""

    def test_y_s_sdag_composition(self):
        """Test that Y, S, and S† work together correctly."""
        from feynman_path.core.gates import GATE_MATRICES

        Y = GATE_MATRICES['Y']
        S = GATE_MATRICES['S']
        Sdag = GATE_MATRICES['Sdag']
        X = GATE_MATRICES['X']

        # Test Y = SXS† relationship
        Y_from_s = S @ X @ Sdag
        assert np.allclose(Y, Y_from_s)

        # Test -X = SYS† relationship
        minus_X = S @ Y @ Sdag
        assert np.allclose(minus_X, -X)

    def test_clifford_group_properties(self):
        """Test that Y, S, Sdag, H form Clifford group properties."""
        from feynman_path.core.gates import GATE_MATRICES

        Y = GATE_MATRICES['Y']
        S = GATE_MATRICES['S']
        H = GATE_MATRICES['H']

        # HYH = -Y
        HYH = H @ Y @ H
        assert np.allclose(HYH, -Y)

    def test_all_new_gates_available_in_registry(self):
        """Test that all new gates are registered."""
        from feynman_path.core.gates import GATE_REGISTRY

        expected_gates = ['y', 'sdag', 't', 'tdag']
        for gate_name in expected_gates:
            assert gate_name in GATE_REGISTRY, f"Gate '{gate_name}' not in registry"
