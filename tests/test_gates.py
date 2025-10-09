"""Tests for Gate class and gate operations."""

import pytest
import numpy as np
import sympy


class TestGateMatrices:
    """Test gate matrix definitions."""

    def test_hadamard_matrix(self):
        """Test Hadamard gate matrix definition."""
        from feynman_path.core.gates import GATE_MATRICES

        H = GATE_MATRICES['H']
        expected = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

        assert np.allclose(H, expected)
        # Test unitarity
        assert np.allclose(H @ H.conj().T, np.eye(2))

    def test_pauli_x_matrix(self):
        """Test Pauli X (NOT) gate matrix."""
        from feynman_path.core.gates import GATE_MATRICES

        X = GATE_MATRICES['X']
        expected = np.array([[0, 1], [1, 0]])

        assert np.allclose(X, expected)
        assert np.allclose(X @ X.conj().T, np.eye(2))

    def test_pauli_z_matrix(self):
        """Test Pauli Z gate matrix."""
        from feynman_path.core.gates import GATE_MATRICES

        Z = GATE_MATRICES['Z']
        expected = np.array([[1, 0], [0, -1]])

        assert np.allclose(Z, expected)
        assert np.allclose(Z @ Z.conj().T, np.eye(2))

    def test_s_gate_matrix(self):
        """Test S (phase) gate matrix."""
        from feynman_path.core.gates import GATE_MATRICES

        S = GATE_MATRICES['S']
        expected = np.array([[1, 0], [0, 1j]])

        assert np.allclose(S, expected)
        assert np.allclose(S @ S.conj().T, np.eye(2))

    def test_cnot_gate_matrix(self):
        """Test CNOT gate matrix."""
        from feynman_path.core.gates import GATE_MATRICES

        CNOT = GATE_MATRICES['CNOT']
        # CNOT in computational basis: |00⟩, |01⟩, |10⟩, |11⟩
        expected = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])

        assert np.allclose(CNOT, expected)
        assert np.allclose(CNOT @ CNOT.conj().T, np.eye(4))


class TestGateClass:
    """Test Gate class functionality."""

    def test_single_qubit_gate_creation(self):
        """Test creating a single-qubit gate."""
        from feynman_path.core.gates import Gate, GATE_MATRICES

        gate = Gate('H', GATE_MATRICES['H'], qubits=(0,), n_qubits=2)

        assert gate.name == 'H'
        assert gate.qubits == (0,)
        assert gate.n_qubits == 2
        assert gate.matrix.shape == (2, 2)

    def test_two_qubit_gate_creation(self):
        """Test creating a two-qubit gate."""
        from feynman_path.core.gates import Gate, GATE_MATRICES

        gate = Gate('CNOT', GATE_MATRICES['CNOT'], qubits=(0, 1), n_qubits=2)

        assert gate.name == 'CNOT'
        assert gate.qubits == (0, 1)
        assert gate.matrix.shape == (4, 4)

    def test_gate_apply_hadamard(self):
        """Test applying Hadamard gate to |0⟩ state."""
        from feynman_path.core.gates import Gate, GATE_MATRICES

        gate = Gate('H', GATE_MATRICES['H'], qubits=(0,), n_qubits=1)

        # Apply H to |0⟩
        transitions = gate.apply_to_state('0')

        # Should give |0⟩ + |1⟩ with equal amplitudes
        assert '0' in transitions
        assert '1' in transitions
        assert abs(transitions['0'] - 1/np.sqrt(2)) < 1e-10
        assert abs(transitions['1'] - 1/np.sqrt(2)) < 1e-10

    def test_gate_apply_x(self):
        """Test applying X gate (bit flip)."""
        from feynman_path.core.gates import Gate, GATE_MATRICES

        gate = Gate('X', GATE_MATRICES['X'], qubits=(0,), n_qubits=1)

        # Apply X to |0⟩ → |1⟩
        transitions = gate.apply_to_state('0')
        assert len(transitions) == 1
        assert '1' in transitions
        assert abs(transitions['1'] - 1.0) < 1e-10

        # Apply X to |1⟩ → |0⟩
        transitions = gate.apply_to_state('1')
        assert len(transitions) == 1
        assert '0' in transitions
        assert abs(transitions['0'] - 1.0) < 1e-10

    def test_gate_apply_z_phase(self):
        """Test applying Z gate (phase flip)."""
        from feynman_path.core.gates import Gate, GATE_MATRICES

        gate = Gate('Z', GATE_MATRICES['Z'], qubits=(0,), n_qubits=1)

        # Apply Z to |0⟩ → |0⟩
        transitions = gate.apply_to_state('0')
        assert '0' in transitions
        assert abs(transitions['0'] - 1.0) < 1e-10

        # Apply Z to |1⟩ → -|1⟩
        transitions = gate.apply_to_state('1')
        assert '1' in transitions
        assert abs(transitions['1'] - (-1.0)) < 1e-10

    def test_gate_apply_cnot(self):
        """Test applying CNOT gate."""
        from feynman_path.core.gates import Gate, GATE_MATRICES

        gate = Gate('CNOT', GATE_MATRICES['CNOT'], qubits=(0, 1), n_qubits=2)

        # CNOT |00⟩ → |00⟩
        transitions = gate.apply_to_state('00')
        assert transitions == {'00': 1.0}

        # CNOT |10⟩ → |11⟩ (control=1, flip target)
        transitions = gate.apply_to_state('10')
        assert transitions == {'11': 1.0}

        # CNOT |01⟩ → |01⟩ (control=0, no flip)
        transitions = gate.apply_to_state('01')
        assert transitions == {'01': 1.0}

        # CNOT |11⟩ → |10⟩
        transitions = gate.apply_to_state('11')
        assert transitions == {'10': 1.0}

    def test_gate_on_specific_qubits(self):
        """Test gate applied to specific qubits in larger system."""
        from feynman_path.core.gates import Gate, GATE_MATRICES

        # H on qubit 1 in a 3-qubit system
        gate = Gate('H', GATE_MATRICES['H'], qubits=(1,), n_qubits=3)

        # Apply H to |010⟩ (q0=0, q1=1, q2=0)
        transitions = gate.apply_to_state('010')

        # Should create superposition on qubit 1: |000⟩ and |010⟩
        # H|1⟩ = (|0⟩ - |1⟩)/√2
        assert '000' in transitions
        assert '010' in transitions
        assert abs(transitions['000'] - 1/np.sqrt(2)) < 1e-10
        assert abs(transitions['010'] - (-1/np.sqrt(2))) < 1e-10


class TestGateRegistry:
    """Test gate registry and factory functions."""

    def test_gate_registry_contains_all_gates(self):
        """Test that GATE_REGISTRY has all standard gates."""
        from feynman_path.core.gates import GATE_REGISTRY

        expected_gates = ['h', 'x', 'z', 's', 'cnot', 'conot']
        for gate_name in expected_gates:
            assert gate_name in GATE_REGISTRY

    def test_create_gate_from_registry_single_qubit(self):
        """Test creating gate from registry."""
        from feynman_path.core.gates import GATE_REGISTRY

        gate = GATE_REGISTRY['h'](qubit=0, n_qubits=2)

        assert gate.name == 'H'
        assert gate.qubits == (0,)
        assert gate.n_qubits == 2

    def test_create_gate_from_registry_two_qubit(self):
        """Test creating two-qubit gate from registry."""
        from feynman_path.core.gates import GATE_REGISTRY

        gate = GATE_REGISTRY['cnot'](control=0, target=1, n_qubits=2)

        assert gate.name == 'CNOT'
        assert gate.qubits == (0, 1)


class TestGateParsing:
    """Test gate specification parsing."""

    def test_parse_single_qubit_gate(self):
        """Test parsing single qubit gate spec like 'h0'."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('h0', n_qubits=2)

        assert gate.name == 'H'
        assert gate.qubits == (0,)

    def test_parse_two_qubit_gate(self):
        """Test parsing two qubit gate spec like 'cnot0,1'."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('cnot0,1', n_qubits=2)

        assert gate.name == 'CNOT'
        assert gate.qubits == (0, 1)

    def test_parse_different_gates(self):
        """Test parsing various gate types."""
        from feynman_path.core.gates import parse_gate_spec

        gates = [
            ('x1', 'X', (1,)),
            ('z2', 'Z', (2,)),
            ('s0', 'S', (0,)),
            ('h3', 'H', (3,)),
        ]

        for spec, expected_name, expected_qubits in gates:
            gate = parse_gate_spec(spec, n_qubits=4)
            assert gate.name == expected_name
            assert gate.qubits == expected_qubits

    def test_parse_invalid_gate_name(self):
        """Test that invalid gate name raises error."""
        from feynman_path.core.gates import parse_gate_spec

        with pytest.raises((ValueError, KeyError)):
            parse_gate_spec('invalid0', n_qubits=2)

    def test_parse_invalid_qubit_index(self):
        """Test that out-of-range qubit index raises error."""
        from feynman_path.core.gates import parse_gate_spec

        with pytest.raises((ValueError, IndexError)):
            parse_gate_spec('h5', n_qubits=2)  # Qubit 5 doesn't exist


class TestCustomGates:
    """Test custom user-defined gates."""

    def test_create_custom_unitary(self):
        """Test creating custom gate from unitary matrix."""
        from feynman_path.core.gates import Gate

        # Custom rotation gate
        theta = np.pi / 4
        custom_matrix = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

        gate = Gate('CustomRot', custom_matrix, qubits=(0,), n_qubits=1)

        assert gate.name == 'CustomRot'
        # Should be unitary
        assert np.allclose(gate.matrix @ gate.matrix.conj().T, np.eye(2))

    def test_validate_unitary(self):
        """Test that non-unitary matrices are rejected."""
        from feynman_path.core.gates import Gate

        # Non-unitary matrix
        bad_matrix = np.array([[1, 0], [0, 2]])  # Not unitary!

        with pytest.raises(ValueError, match="unitary"):
            Gate('BadGate', bad_matrix, qubits=(0,), n_qubits=1, validate=True)


class TestThreeQubitCircuits:
    """Test gates on 3-qubit circuits with various qubit orderings."""

    def test_cnot_2_0_on_three_qubits(self):
        """Test CNOT with control=2, target=0 on 3-qubit system."""
        from feynman_path.core.gates import Gate, GATE_MATRICES

        gate = Gate('CNOT', GATE_MATRICES['CNOT'], qubits=(2, 0), n_qubits=3)

        # Convention: bitstring[i] = qubit i, so '001' means q0=0, q1=0, q2=1

        # Test |001⟩ (q0=0, q1=0, q2=1) → |101⟩ (control q2=1, flip q0: 0→1)
        transitions = gate.apply_to_state('001')
        assert transitions == {'101': 1.0}

        # Test |101⟩ (q0=1, q1=0, q2=1) → |001⟩ (control q2=1, flip q0: 1→0)
        transitions = gate.apply_to_state('101')
        assert transitions == {'001': 1.0}

        # Test |100⟩ (q0=0, q1=0, q2=1) → |100⟩ no change, but wait - q2=1 so should flip!
        # Actually '100' means q0=1, q1=0, q2=0, so control q2=0, no flip
        transitions = gate.apply_to_state('100')
        assert transitions == {'100': 1.0}

        # Test |010⟩ (q0=0, q1=1, q2=0) → no flip (control q2=0)
        transitions = gate.apply_to_state('010')
        assert transitions == {'010': 1.0}

        # Test |011⟩ (q0=0, q1=1, q2=1) → |111⟩ (control q2=1, flip q0: 0→1)
        transitions = gate.apply_to_state('011')
        assert transitions == {'111': 1.0}

    def test_cnot_0_2_on_three_qubits(self):
        """Test CNOT with control=0, target=2 on 3-qubit system."""
        from feynman_path.core.gates import Gate, GATE_MATRICES

        gate = Gate('CNOT', GATE_MATRICES['CNOT'], qubits=(0, 2), n_qubits=3)

        # Convention: '001' means q0=0, q1=0, q2=1

        # Test |100⟩ (q0=1, q1=0, q2=0) → |101⟩ (control q0=1, flip q2: 0→1)
        transitions = gate.apply_to_state('100')
        assert transitions == {'101': 1.0}

        # Test |000⟩ (q0=0, q1=0, q2=0) → |000⟩ (control q0=0, no flip)
        transitions = gate.apply_to_state('000')
        assert transitions == {'000': 1.0}

        # Test |010⟩ (q0=0, q1=1, q2=0) → |010⟩ (control q0=0, no flip)
        transitions = gate.apply_to_state('010')
        assert transitions == {'010': 1.0}

        # Test |111⟩ (q0=1, q1=1, q2=1) → |110⟩ (control q0=1, flip q2: 1→0)
        transitions = gate.apply_to_state('111')
        assert transitions == {'110': 1.0}

    def test_cnot_1_2_on_three_qubits(self):
        """Test CNOT with control=1, target=2 on 3-qubit system."""
        from feynman_path.core.gates import Gate, GATE_MATRICES

        gate = Gate('CNOT', GATE_MATRICES['CNOT'], qubits=(1, 2), n_qubits=3)

        # Test |010⟩ (q0=0, q1=1, q2=0) → |011⟩ (control q1=1, flip q2: 0→1)
        transitions = gate.apply_to_state('010')
        assert transitions == {'011': 1.0}

        # Test |011⟩ (q0=0, q1=1, q2=1) → |010⟩ (control q1=1, flip q2: 1→0)
        transitions = gate.apply_to_state('011')
        assert transitions == {'010': 1.0}

        # Test |000⟩ (q0=0, q1=0, q2=0) → |000⟩ (control q1=0, no flip)
        transitions = gate.apply_to_state('000')
        assert transitions == {'000': 1.0}

    def test_hadamard_on_qubit_1_of_three(self):
        """Test Hadamard on middle qubit (q1) in 3-qubit system."""
        from feynman_path.core.gates import Gate, GATE_MATRICES

        gate = Gate('H', GATE_MATRICES['H'], qubits=(1,), n_qubits=3)

        # Test |010⟩ (q0=0, q1=1, q2=0) → superposition
        transitions = gate.apply_to_state('010')

        assert '000' in transitions
        assert '010' in transitions
        # H|1⟩ = (|0⟩ - |1⟩)/√2
        assert abs(transitions['000'] - 1/np.sqrt(2)) < 1e-10
        assert abs(transitions['010'] - (-1/np.sqrt(2))) < 1e-10

    def test_hadamard_on_qubit_2_of_three(self):
        """Test Hadamard on highest qubit (q2) in 3-qubit system."""
        from feynman_path.core.gates import Gate, GATE_MATRICES

        gate = Gate('H', GATE_MATRICES['H'], qubits=(2,), n_qubits=3)

        # Test |011⟩ (q0=0, q1=1, q2=1) → superposition on q2
        # Convention: bitstring[i] = qubit i value
        transitions = gate.apply_to_state('011')

        assert '010' in transitions  # q2: 1 → 0
        assert '011' in transitions  # q2: 1 → 1
        # H|1⟩ = (|0⟩ - |1⟩)/√2
        assert abs(transitions['010'] - 1/np.sqrt(2)) < 1e-10
        assert abs(transitions['011'] - (-1/np.sqrt(2))) < 1e-10

    def test_x_gate_on_specific_qubit(self):
        """Test X gate flips only the target qubit in 3-qubit system."""
        from feynman_path.core.gates import Gate, GATE_MATRICES

        # X on qubit 1
        gate = Gate('X', GATE_MATRICES['X'], qubits=(1,), n_qubits=3)

        # Test |101⟩ → |111⟩ (flip q1: 0→1)
        transitions = gate.apply_to_state('101')
        assert transitions == {'111': 1.0}

        # Test |111⟩ → |101⟩ (flip q1: 1→0)
        transitions = gate.apply_to_state('111')
        assert transitions == {'101': 1.0}

    def test_z_gate_on_middle_qubit(self):
        """Test Z gate phase flip on middle qubit."""
        from feynman_path.core.gates import Gate, GATE_MATRICES

        gate = Gate('Z', GATE_MATRICES['Z'], qubits=(1,), n_qubits=3)

        # Test |010⟩ → -|010⟩ (q1=1, gets negative phase)
        transitions = gate.apply_to_state('010')
        assert '010' in transitions
        assert abs(transitions['010'] - (-1.0)) < 1e-10

        # Test |000⟩ → |000⟩ (q1=0, no phase change)
        transitions = gate.apply_to_state('000')
        assert transitions == {'000': 1.0}

    def test_sequential_gates_on_different_qubits(self):
        """Test applying multiple gates in sequence on different qubits."""
        from feynman_path.core.gates import Gate, GATE_MATRICES

        # Start with |000⟩ (q0=0, q1=0, q2=0)
        # Apply H on q0
        h0 = Gate('H', GATE_MATRICES['H'], qubits=(0,), n_qubits=3)
        state1 = h0.apply_to_state('000')
        # Should get |000⟩ + |100⟩ (q0 in superposition)
        assert '000' in state1
        assert '100' in state1

        # Apply CNOT(0,1) to |100⟩ result
        cnot01 = Gate('CNOT', GATE_MATRICES['CNOT'], qubits=(0, 1), n_qubits=3)
        state2 = cnot01.apply_to_state('100')
        # q0=1, so flip q1: |100⟩ → |110⟩
        assert state2 == {'110': 1.0}

    def test_non_adjacent_qubits_cnot(self):
        """Test CNOT on non-adjacent qubits (skipping middle qubit)."""
        from feynman_path.core.gates import Gate, GATE_MATRICES

        # CNOT control=0, target=2 (skips qubit 1)
        gate = Gate('CNOT', GATE_MATRICES['CNOT'], qubits=(0, 2), n_qubits=3)

        # Test |110⟩ (q0=1, q1=1, q2=0)
        # Control q0=1, so flip q2: 0→1
        # Middle qubit q1 should be unchanged
        transitions = gate.apply_to_state('110')
        assert transitions == {'111': 1.0}
