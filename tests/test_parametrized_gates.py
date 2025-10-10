"""Tests for parametrized rotation gates: Rx, Ry, Rz."""

import pytest
import numpy as np
import sympy


class TestRxGate:
    """Test Rx (X-rotation) gate."""

    def test_rx_matrix_generation_zero_angle(self):
        """Test Rx(0) = I."""
        from feynman_path.core.gates import generate_rx_matrix

        theta = 0
        Rx = generate_rx_matrix(theta)
        I = np.eye(2, dtype=np.complex128)

        assert np.allclose(Rx, I)

    def test_rx_matrix_generation_pi(self):
        """Test Rx(π) = -iX."""
        from feynman_path.core.gates import generate_rx_matrix, GATE_MATRICES

        theta = np.pi
        Rx = generate_rx_matrix(theta)
        X = GATE_MATRICES['X']
        minus_i_X = -1j * X

        assert np.allclose(Rx, minus_i_X)

    def test_rx_matrix_generation_pi_over_2(self):
        """Test Rx(π/2) matrix."""
        from feynman_path.core.gates import generate_rx_matrix

        theta = np.pi / 2
        Rx = generate_rx_matrix(theta)

        # Rx(π/2) = (1/√2) * [[1, -i], [-i, 1]]
        expected = (1/np.sqrt(2)) * np.array([[1, -1j], [-1j, 1]], dtype=np.complex128)

        assert np.allclose(Rx, expected)

    def test_rx_is_unitary(self):
        """Test that Rx(θ) is unitary for various angles."""
        from feynman_path.core.gates import generate_rx_matrix

        angles = [0, np.pi/4, np.pi/2, np.pi, 2*np.pi, -np.pi/3]

        for theta in angles:
            Rx = generate_rx_matrix(theta)
            assert np.allclose(Rx @ Rx.conj().T, np.eye(2))

    def test_rx_matrix_form(self):
        """Test Rx(θ) = cos(θ/2)*I - i*sin(θ/2)*X."""
        from feynman_path.core.gates import generate_rx_matrix, GATE_MATRICES

        theta = np.pi / 3
        Rx = generate_rx_matrix(theta)

        I = np.eye(2, dtype=np.complex128)
        X = GATE_MATRICES['X']
        expected = np.cos(theta/2) * I - 1j * np.sin(theta/2) * X

        assert np.allclose(Rx, expected)

    def test_rx_composition(self):
        """Test Rx(θ₁)Rx(θ₂) = Rx(θ₁+θ₂)."""
        from feynman_path.core.gates import generate_rx_matrix

        theta1 = np.pi / 4
        theta2 = np.pi / 6

        Rx1 = generate_rx_matrix(theta1)
        Rx2 = generate_rx_matrix(theta2)
        Rx_combined = generate_rx_matrix(theta1 + theta2)

        assert np.allclose(Rx1 @ Rx2, Rx_combined, atol=1e-10)

    def test_rx_two_pi_is_minus_identity(self):
        """Test Rx(2π) = -I (global phase)."""
        from feynman_path.core.gates import generate_rx_matrix

        theta = 2 * np.pi
        Rx = generate_rx_matrix(theta)
        minus_I = -np.eye(2, dtype=np.complex128)

        assert np.allclose(Rx, minus_I)

    def test_parse_rx_gate_with_numeric_angle(self):
        """Test parsing Rx gate with numeric angle."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('rx0,1.5708', n_qubits=2)  # π/2 ≈ 1.5708

        assert gate.name == 'Rx'
        assert gate.qubits == (0,)
        assert hasattr(gate, 'angle')
        assert abs(gate.angle - np.pi/2) < 0.001

    def test_parse_rx_gate_with_pi(self):
        """Test parsing Rx gate with 'pi' string."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('rx1,pi', n_qubits=2)

        assert gate.name == 'Rx'
        assert gate.qubits == (1,)
        assert abs(gate.angle - np.pi) < 1e-10

    def test_parse_rx_gate_with_pi_expression(self):
        """Test parsing Rx gate with 'pi/4' expression."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('rx0,pi/4', n_qubits=3)

        assert gate.name == 'Rx'
        assert gate.qubits == (0,)
        assert abs(gate.angle - np.pi/4) < 1e-10

    def test_rx_apply_to_zero_state_pi_over_2(self):
        """Test Rx(π/2)|0⟩ creates superposition."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('rx0,pi/2', n_qubits=1)
        transitions = gate.apply_to_state('0')

        # Rx(π/2)|0⟩ = (1/√2)(|0⟩ - i|1⟩)
        assert '0' in transitions
        assert '1' in transitions
        assert abs(transitions['0'] - 1/np.sqrt(2)) < 1e-10
        assert abs(transitions['1'] - (-1j/np.sqrt(2))) < 1e-10

    def test_rx_apply_to_one_state_pi(self):
        """Test Rx(π)|1⟩ = -i|0⟩."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('rx0,pi', n_qubits=1)
        transitions = gate.apply_to_state('1')

        assert len(transitions) == 1
        assert '0' in transitions
        assert abs(transitions['0'] - (-1j)) < 1e-10

    def test_rx_in_multi_qubit_system(self):
        """Test Rx on specific qubit in multi-qubit system."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('rx1,pi/2', n_qubits=3)

        # Apply to |010⟩ (q1=1)
        transitions = gate.apply_to_state('010')

        # Should create superposition on q1
        assert '000' in transitions
        assert '010' in transitions


class TestRyGate:
    """Test Ry (Y-rotation) gate."""

    def test_ry_matrix_generation_zero_angle(self):
        """Test Ry(0) = I."""
        from feynman_path.core.gates import generate_ry_matrix

        theta = 0
        Ry = generate_ry_matrix(theta)
        I = np.eye(2, dtype=np.complex128)

        assert np.allclose(Ry, I)

    def test_ry_matrix_generation_pi(self):
        """Test Ry(π) = -iY."""
        from feynman_path.core.gates import generate_ry_matrix, GATE_MATRICES

        theta = np.pi
        Ry = generate_ry_matrix(theta)
        Y = GATE_MATRICES['Y']
        minus_i_Y = -1j * Y

        assert np.allclose(Ry, minus_i_Y)

    def test_ry_matrix_generation_pi_over_2(self):
        """Test Ry(π/2) matrix."""
        from feynman_path.core.gates import generate_ry_matrix

        theta = np.pi / 2
        Ry = generate_ry_matrix(theta)

        # Ry(π/2) = (1/√2) * [[1, -1], [1, 1]]
        expected = (1/np.sqrt(2)) * np.array([[1, -1], [1, 1]], dtype=np.complex128)

        assert np.allclose(Ry, expected)

    def test_ry_is_unitary(self):
        """Test that Ry(θ) is unitary for various angles."""
        from feynman_path.core.gates import generate_ry_matrix

        angles = [0, np.pi/4, np.pi/2, np.pi, 2*np.pi, -np.pi/3]

        for theta in angles:
            Ry = generate_ry_matrix(theta)
            assert np.allclose(Ry @ Ry.conj().T, np.eye(2))

    def test_ry_matrix_form(self):
        """Test Ry(θ) = cos(θ/2)*I - i*sin(θ/2)*Y."""
        from feynman_path.core.gates import generate_ry_matrix, GATE_MATRICES

        theta = np.pi / 3
        Ry = generate_ry_matrix(theta)

        I = np.eye(2, dtype=np.complex128)
        Y = GATE_MATRICES['Y']
        expected = np.cos(theta/2) * I - 1j * np.sin(theta/2) * Y

        assert np.allclose(Ry, expected)

    def test_ry_composition(self):
        """Test Ry(θ₁)Ry(θ₂) = Ry(θ₁+θ₂)."""
        from feynman_path.core.gates import generate_ry_matrix

        theta1 = np.pi / 4
        theta2 = np.pi / 6

        Ry1 = generate_ry_matrix(theta1)
        Ry2 = generate_ry_matrix(theta2)
        Ry_combined = generate_ry_matrix(theta1 + theta2)

        assert np.allclose(Ry1 @ Ry2, Ry_combined, atol=1e-10)

    def test_parse_ry_gate_with_numeric_angle(self):
        """Test parsing Ry gate with numeric angle."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('ry0,0.7854', n_qubits=2)  # π/4 ≈ 0.7854

        assert gate.name == 'Ry'
        assert gate.qubits == (0,)
        assert abs(gate.angle - np.pi/4) < 0.001

    def test_parse_ry_gate_with_pi_expression(self):
        """Test parsing Ry gate with 'pi/2' expression."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('ry1,pi/2', n_qubits=2)

        assert gate.name == 'Ry'
        assert gate.qubits == (1,)
        assert abs(gate.angle - np.pi/2) < 1e-10

    def test_ry_apply_to_zero_state_pi_over_2(self):
        """Test Ry(π/2)|0⟩ = (|0⟩ + |1⟩)/√2."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('ry0,pi/2', n_qubits=1)
        transitions = gate.apply_to_state('0')

        # Ry(π/2)|0⟩ = (1/√2)(|0⟩ + |1⟩)
        assert '0' in transitions
        assert '1' in transitions
        assert abs(transitions['0'] - 1/np.sqrt(2)) < 1e-10
        assert abs(transitions['1'] - 1/np.sqrt(2)) < 1e-10


class TestRzGate:
    """Test Rz (Z-rotation) gate."""

    def test_rz_matrix_generation_zero_angle(self):
        """Test Rz(0) = I."""
        from feynman_path.core.gates import generate_rz_matrix

        theta = 0
        Rz = generate_rz_matrix(theta)
        I = np.eye(2, dtype=np.complex128)

        assert np.allclose(Rz, I)

    def test_rz_matrix_generation_pi(self):
        """Test Rz(π) = -iZ."""
        from feynman_path.core.gates import generate_rz_matrix, GATE_MATRICES

        theta = np.pi
        Rz = generate_rz_matrix(theta)
        Z = GATE_MATRICES['Z']
        minus_i_Z = -1j * Z

        assert np.allclose(Rz, minus_i_Z)

    def test_rz_matrix_generation_pi_over_2(self):
        """Test Rz(π/2) matrix."""
        from feynman_path.core.gates import generate_rz_matrix

        theta = np.pi / 2
        Rz = generate_rz_matrix(theta)

        # Rz(π/2) = [[e^(-iπ/4), 0], [0, e^(iπ/4)]]
        expected = np.array([
            [np.exp(-1j*np.pi/4), 0],
            [0, np.exp(1j*np.pi/4)]
        ], dtype=np.complex128)

        assert np.allclose(Rz, expected)

    def test_rz_is_unitary(self):
        """Test that Rz(θ) is unitary for various angles."""
        from feynman_path.core.gates import generate_rz_matrix

        angles = [0, np.pi/4, np.pi/2, np.pi, 2*np.pi, -np.pi/3]

        for theta in angles:
            Rz = generate_rz_matrix(theta)
            assert np.allclose(Rz @ Rz.conj().T, np.eye(2))

    def test_rz_matrix_form(self):
        """Test Rz(θ) = cos(θ/2)*I - i*sin(θ/2)*Z."""
        from feynman_path.core.gates import generate_rz_matrix, GATE_MATRICES

        theta = np.pi / 3
        Rz = generate_rz_matrix(theta)

        I = np.eye(2, dtype=np.complex128)
        Z = GATE_MATRICES['Z']
        expected = np.cos(theta/2) * I - 1j * np.sin(theta/2) * Z

        assert np.allclose(Rz, expected)

    def test_rz_composition(self):
        """Test Rz(θ₁)Rz(θ₂) = Rz(θ₁+θ₂)."""
        from feynman_path.core.gates import generate_rz_matrix

        theta1 = np.pi / 4
        theta2 = np.pi / 6

        Rz1 = generate_rz_matrix(theta1)
        Rz2 = generate_rz_matrix(theta2)
        Rz_combined = generate_rz_matrix(theta1 + theta2)

        assert np.allclose(Rz1 @ Rz2, Rz_combined, atol=1e-10)

    def test_parse_rz_gate_with_pi_expression(self):
        """Test parsing Rz gate with 'pi/4' expression."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('rz2,pi/4', n_qubits=3)

        assert gate.name == 'Rz'
        assert gate.qubits == (2,)
        assert abs(gate.angle - np.pi/4) < 1e-10

    def test_rz_apply_to_zero_state(self):
        """Test Rz(θ)|0⟩ = e^(-iθ/2)|0⟩."""
        from feynman_path.core.gates import parse_gate_spec

        theta = np.pi / 3
        gate = parse_gate_spec(f'rz0,{theta}', n_qubits=1)
        transitions = gate.apply_to_state('0')

        expected_phase = np.exp(-1j * theta / 2)
        assert len(transitions) == 1
        assert '0' in transitions
        assert abs(transitions['0'] - expected_phase) < 1e-10

    def test_rz_apply_to_one_state(self):
        """Test Rz(θ)|1⟩ = e^(iθ/2)|1⟩."""
        from feynman_path.core.gates import parse_gate_spec

        theta = np.pi / 3
        gate = parse_gate_spec(f'rz0,{theta}', n_qubits=1)
        transitions = gate.apply_to_state('1')

        expected_phase = np.exp(1j * theta / 2)
        assert len(transitions) == 1
        assert '1' in transitions
        assert abs(transitions['1'] - expected_phase) < 1e-10


class TestRotationGatesIntegration:
    """Integration tests for rotation gates."""

    def test_all_rotation_gates_in_registry(self):
        """Test that all rotation gates are registered."""
        from feynman_path.core.gates import GATE_REGISTRY

        # Note: Rotation gates may not be in registry if they require special parsing
        # This test will be updated based on implementation approach
        pass

    def test_rx_ry_rz_different_results(self):
        """Test that Rx, Ry, Rz produce different results."""
        from feynman_path.core.gates import generate_rx_matrix, generate_ry_matrix, generate_rz_matrix

        theta = np.pi / 4
        Rx = generate_rx_matrix(theta)
        Ry = generate_ry_matrix(theta)
        Rz = generate_rz_matrix(theta)

        # All should be different
        assert not np.allclose(Rx, Ry)
        assert not np.allclose(Rx, Rz)
        assert not np.allclose(Ry, Rz)

    def test_rotation_gate_with_sympy_angle(self):
        """Test rotation gate with symbolic angle."""
        from feynman_path.core.gates import parse_gate_spec

        # Test with '2*pi/3' expression
        gate = parse_gate_spec('rx0,2*pi/3', n_qubits=2)

        assert gate.name == 'Rx'
        assert abs(gate.angle - 2*np.pi/3) < 1e-10

    def test_negative_angle(self):
        """Test rotation gates with negative angles."""
        from feynman_path.core.gates import generate_rx_matrix

        theta = -np.pi / 4
        Rx_neg = generate_rx_matrix(theta)
        Rx_pos = generate_rx_matrix(-theta)

        # Rx(-θ) = Rx(θ)† (inverse rotation)
        assert np.allclose(Rx_neg, Rx_pos.conj().T)

    def test_parse_multiple_rotation_gates_in_circuit(self):
        """Test parsing multiple rotation gates in a circuit."""
        from feynman_path.core.gates import parse_gate_spec

        gates_specs = ['rx0,pi/4', 'ry1,pi/2', 'rz0,pi']

        for spec in gates_specs:
            gate = parse_gate_spec(spec, n_qubits=2)
            assert gate is not None
            assert hasattr(gate, 'angle')
