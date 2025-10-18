"""Tests for controlled rotation gates: CRx, CRy, CRz."""

import pytest
import numpy as np


class TestCRxGate:
    """Test CRx (controlled X-rotation) gate."""

    def test_crx_matrix_generation_zero_angle(self):
        """Test CRx(0) = I (4x4 identity)."""
        from feynman_path.core.gates import generate_crx_matrix

        theta = 0
        CRx = generate_crx_matrix(theta)
        I4 = np.eye(4, dtype=np.complex128)

        assert np.allclose(CRx, I4)

    def test_crx_matrix_generation_pi(self):
        """Test CRx(π) matrix structure."""
        from feynman_path.core.gates import generate_crx_matrix

        theta = np.pi
        CRx = generate_crx_matrix(theta)

        # CRx(π) should have I in top-left 2x2 block and -iX in bottom-right
        # Upper 2x2: identity
        assert np.allclose(CRx[0:2, 0:2], np.eye(2))
        # Lower 2x2: -iX = [[0, -i], [-i, 0]]
        expected_bottom_right = np.array([[0, -1j], [-1j, 0]], dtype=np.complex128)
        assert np.allclose(CRx[2:4, 2:4], expected_bottom_right)
        # Off-diagonal blocks should be zero
        assert np.allclose(CRx[0:2, 2:4], 0)
        assert np.allclose(CRx[2:4, 0:2], 0)

    def test_crx_matrix_generation_pi_over_2(self):
        """Test CRx(π/2) matrix explicit values."""
        from feynman_path.core.gates import generate_crx_matrix

        theta = np.pi / 2
        CRx = generate_crx_matrix(theta)

        # CRx(π/2) structure: [I₂, 0; 0, Rx(π/2)]
        # Rx(π/2) = (1/√2)[[1, -i], [-i, 1]]
        sqrt2_inv = 1 / np.sqrt(2)
        expected = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, sqrt2_inv, -1j*sqrt2_inv],
            [0, 0, -1j*sqrt2_inv, sqrt2_inv]
        ], dtype=np.complex128)

        assert np.allclose(CRx, expected)

    def test_crx_is_unitary(self):
        """Test that CRx(θ) is unitary for various angles."""
        from feynman_path.core.gates import generate_crx_matrix

        angles = [0, np.pi/4, np.pi/2, np.pi, 2*np.pi, -np.pi/3]

        for theta in angles:
            CRx = generate_crx_matrix(theta)
            # Check U†U = I
            product = CRx @ CRx.conj().T
            assert np.allclose(product, np.eye(4), atol=1e-10)

    def test_crx_matrix_structure(self):
        """Test CRx = |0⟩⟨0|⊗I + |1⟩⟨1|⊗Rx(θ)."""
        from feynman_path.core.gates import generate_crx_matrix, generate_rx_matrix

        theta = np.pi / 3
        CRx = generate_crx_matrix(theta)
        Rx = generate_rx_matrix(theta)

        # Top-left 2x2 block should be identity
        assert np.allclose(CRx[0:2, 0:2], np.eye(2))
        # Bottom-right 2x2 block should be Rx(θ)
        assert np.allclose(CRx[2:4, 2:4], Rx)
        # Off-diagonal blocks should be zero
        assert np.allclose(CRx[0:2, 2:4], 0)
        assert np.allclose(CRx[2:4, 0:2], 0)

    def test_crx_control_zero_unchanged(self):
        """Test that when control qubit is 0, target is unchanged."""
        from feynman_path.core.gates import generate_crx_matrix

        theta = np.pi / 4
        CRx = generate_crx_matrix(theta)

        # For |00⟩ (index 0) and |01⟩ (index 1), should get identity operation
        # CRx|00⟩ = |00⟩
        input_00 = np.array([1, 0, 0, 0], dtype=np.complex128)
        output_00 = CRx @ input_00
        assert np.allclose(output_00, input_00)

        # CRx|01⟩ = |01⟩
        input_01 = np.array([0, 1, 0, 0], dtype=np.complex128)
        output_01 = CRx @ input_01
        assert np.allclose(output_01, input_01)

    def test_crx_control_one_applies_rx(self):
        """Test that when control qubit is 1, Rx is applied to target."""
        from feynman_path.core.gates import generate_crx_matrix, generate_rx_matrix

        theta = np.pi / 2
        CRx = generate_crx_matrix(theta)
        Rx = generate_rx_matrix(theta)

        # CRx|10⟩ should apply Rx to target qubit
        # |10⟩ in our basis is index 2 (control=1, target=0)
        input_10 = np.array([0, 0, 1, 0], dtype=np.complex128)
        output = CRx @ input_10

        # Expected: Rx applied to target creates superposition
        # Rx(π/2)|0⟩ = (1/√2)(|0⟩ - i|1⟩)
        # So CRx|10⟩ = (1/√2)(|10⟩ - i|11⟩)
        sqrt2_inv = 1 / np.sqrt(2)
        expected = np.array([0, 0, sqrt2_inv, -1j*sqrt2_inv], dtype=np.complex128)
        assert np.allclose(output, expected)


class TestCRyGate:
    """Test CRy (controlled Y-rotation) gate."""

    def test_cry_matrix_generation_zero_angle(self):
        """Test CRy(0) = I (4x4 identity)."""
        from feynman_path.core.gates import generate_cry_matrix

        theta = 0
        CRy = generate_cry_matrix(theta)
        I4 = np.eye(4, dtype=np.complex128)

        assert np.allclose(CRy, I4)

    def test_cry_matrix_generation_pi_over_2(self):
        """Test CRy(π/2) matrix explicit values."""
        from feynman_path.core.gates import generate_cry_matrix

        theta = np.pi / 2
        CRy = generate_cry_matrix(theta)

        # CRy(π/2) structure: [I₂, 0; 0, Ry(π/2)]
        # Ry(π/2) = (1/√2)[[1, -1], [1, 1]]
        sqrt2_inv = 1 / np.sqrt(2)
        expected = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, sqrt2_inv, -sqrt2_inv],
            [0, 0, sqrt2_inv, sqrt2_inv]
        ], dtype=np.complex128)

        assert np.allclose(CRy, expected)

    def test_cry_is_unitary(self):
        """Test that CRy(θ) is unitary for various angles."""
        from feynman_path.core.gates import generate_cry_matrix

        angles = [0, np.pi/4, np.pi/2, np.pi, 2*np.pi, -np.pi/3]

        for theta in angles:
            CRy = generate_cry_matrix(theta)
            # Check U†U = I
            product = CRy @ CRy.conj().T
            assert np.allclose(product, np.eye(4), atol=1e-10)

    def test_cry_matrix_structure(self):
        """Test CRy = |0⟩⟨0|⊗I + |1⟩⟨1|⊗Ry(θ)."""
        from feynman_path.core.gates import generate_cry_matrix, generate_ry_matrix

        theta = np.pi / 3
        CRy = generate_cry_matrix(theta)
        Ry = generate_ry_matrix(theta)

        # Top-left 2x2 block should be identity
        assert np.allclose(CRy[0:2, 0:2], np.eye(2))
        # Bottom-right 2x2 block should be Ry(θ)
        assert np.allclose(CRy[2:4, 2:4], Ry)
        # Off-diagonal blocks should be zero
        assert np.allclose(CRy[0:2, 2:4], 0)
        assert np.allclose(CRy[2:4, 0:2], 0)

    def test_cry_control_zero_unchanged(self):
        """Test that when control qubit is 0, target is unchanged."""
        from feynman_path.core.gates import generate_cry_matrix

        theta = np.pi / 4
        CRy = generate_cry_matrix(theta)

        # CRy|00⟩ = |00⟩
        input_00 = np.array([1, 0, 0, 0], dtype=np.complex128)
        output_00 = CRy @ input_00
        assert np.allclose(output_00, input_00)

        # CRy|01⟩ = |01⟩
        input_01 = np.array([0, 1, 0, 0], dtype=np.complex128)
        output_01 = CRy @ input_01
        assert np.allclose(output_01, input_01)

    def test_cry_control_one_applies_ry(self):
        """Test that when control qubit is 1, Ry is applied to target."""
        from feynman_path.core.gates import generate_cry_matrix, generate_ry_matrix

        theta = np.pi / 2
        CRy = generate_cry_matrix(theta)
        Ry = generate_ry_matrix(theta)

        # CRy|10⟩ should apply Ry to target qubit
        input_10 = np.array([0, 0, 1, 0], dtype=np.complex128)
        output = CRy @ input_10

        # Ry(π/2)|0⟩ = (1/√2)(|0⟩ + |1⟩)
        # So CRy|10⟩ = (1/√2)(|10⟩ + |11⟩)
        sqrt2_inv = 1 / np.sqrt(2)
        expected = np.array([0, 0, sqrt2_inv, sqrt2_inv], dtype=np.complex128)
        assert np.allclose(output, expected)


class TestCRzGate:
    """Test CRz (controlled Z-rotation) gate."""

    def test_crz_matrix_generation_zero_angle(self):
        """Test CRz(0) = I (4x4 identity)."""
        from feynman_path.core.gates import generate_crz_matrix

        theta = 0
        CRz = generate_crz_matrix(theta)
        I4 = np.eye(4, dtype=np.complex128)

        assert np.allclose(CRz, I4)

    def test_crz_matrix_generation_pi_over_4(self):
        """Test CRz(π/4) matrix explicit values."""
        from feynman_path.core.gates import generate_crz_matrix

        theta = np.pi / 4
        CRz = generate_crz_matrix(theta)

        # CRz(π/4) structure: [I₂, 0; 0, Rz(π/4)]
        # Rz(π/4) = [[e^(-iπ/8), 0], [0, e^(iπ/8)]]
        expected = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, np.exp(-1j*np.pi/8), 0],
            [0, 0, 0, np.exp(1j*np.pi/8)]
        ], dtype=np.complex128)

        assert np.allclose(CRz, expected)

    def test_crz_is_unitary(self):
        """Test that CRz(θ) is unitary for various angles."""
        from feynman_path.core.gates import generate_crz_matrix

        angles = [0, np.pi/4, np.pi/2, np.pi, 2*np.pi, -np.pi/3]

        for theta in angles:
            CRz = generate_crz_matrix(theta)
            # Check U†U = I
            product = CRz @ CRz.conj().T
            assert np.allclose(product, np.eye(4), atol=1e-10)

    def test_crz_matrix_structure(self):
        """Test CRz = |0⟩⟨0|⊗I + |1⟩⟨1|⊗Rz(θ)."""
        from feynman_path.core.gates import generate_crz_matrix, generate_rz_matrix

        theta = np.pi / 3
        CRz = generate_crz_matrix(theta)
        Rz = generate_rz_matrix(theta)

        # Top-left 2x2 block should be identity
        assert np.allclose(CRz[0:2, 0:2], np.eye(2))
        # Bottom-right 2x2 block should be Rz(θ)
        assert np.allclose(CRz[2:4, 2:4], Rz)
        # Off-diagonal blocks should be zero
        assert np.allclose(CRz[0:2, 2:4], 0)
        assert np.allclose(CRz[2:4, 0:2], 0)

    def test_crz_control_zero_unchanged(self):
        """Test that when control qubit is 0, target is unchanged."""
        from feynman_path.core.gates import generate_crz_matrix

        theta = np.pi / 4
        CRz = generate_crz_matrix(theta)

        # CRz|00⟩ = |00⟩
        input_00 = np.array([1, 0, 0, 0], dtype=np.complex128)
        output_00 = CRz @ input_00
        assert np.allclose(output_00, input_00)

        # CRz|01⟩ = |01⟩
        input_01 = np.array([0, 1, 0, 0], dtype=np.complex128)
        output_01 = CRz @ input_01
        assert np.allclose(output_01, input_01)

    def test_crz_control_one_applies_rz(self):
        """Test that when control qubit is 1, Rz is applied to target."""
        from feynman_path.core.gates import generate_crz_matrix

        theta = np.pi / 2
        CRz = generate_crz_matrix(theta)

        # CRz|10⟩ should apply Rz to target qubit
        # Rz only adds phase, doesn't change state
        input_10 = np.array([0, 0, 1, 0], dtype=np.complex128)
        output = CRz @ input_10

        # Rz(π/2)|0⟩ = e^(-iπ/4)|0⟩
        # So CRz|10⟩ = e^(-iπ/4)|10⟩
        expected_phase = np.exp(-1j * np.pi / 4)
        expected = np.array([0, 0, expected_phase, 0], dtype=np.complex128)
        assert np.allclose(output, expected)

    def test_crz_diagonal_structure(self):
        """Test that CRz is diagonal (phase-only gate)."""
        from feynman_path.core.gates import generate_crz_matrix

        theta = np.pi / 3
        CRz = generate_crz_matrix(theta)

        # CRz should be diagonal
        # Check off-diagonal elements are zero
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert abs(CRz[i, j]) < 1e-10


class TestParsingControlledRotationGates:
    """Test parsing of controlled rotation gate specifications."""

    def test_parse_crx_gate_with_numeric_angle(self):
        """Test parsing CRx gate with numeric angle."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('crx0,1,1.5708', n_qubits=3)

        assert gate.name == 'CRx'
        assert gate.qubits == (0, 1)
        assert hasattr(gate, 'angle')
        assert abs(gate.angle - 1.5708) < 0.0001

    def test_parse_crx_gate_with_pi(self):
        """Test parsing CRx gate with 'pi' string."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('crx1,2,pi', n_qubits=3)

        assert gate.name == 'CRx'
        assert gate.qubits == (1, 2)
        assert abs(gate.angle - np.pi) < 1e-10

    def test_parse_crx_gate_with_pi_expression(self):
        """Test parsing CRx gate with 'pi/4' expression."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('crx0,1,pi/4', n_qubits=2)

        assert gate.name == 'CRx'
        assert gate.qubits == (0, 1)
        assert abs(gate.angle - np.pi/4) < 1e-10

    def test_parse_cry_gate_basic(self):
        """Test parsing CRy gate with numeric angle."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('cry1,2,1.5708', n_qubits=3)

        assert gate.name == 'CRy'
        assert gate.qubits == (1, 2)
        assert abs(gate.angle - 1.5708) < 0.0001

    def test_parse_crz_gate_basic(self):
        """Test parsing CRz gate with pi/4 expression."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('crz0,1,pi/4', n_qubits=2)

        assert gate.name == 'CRz'
        assert gate.qubits == (0, 1)
        assert abs(gate.angle - np.pi/4) < 1e-10

    def test_parse_controlled_rotation_invalid_format(self):
        """Test that invalid format raises error."""
        from feynman_path.core.gates import parse_gate_spec

        # Missing angle
        with pytest.raises(ValueError, match="requires format"):
            parse_gate_spec('crx0,1', n_qubits=2)

        # Missing target
        with pytest.raises(ValueError, match="requires format"):
            parse_gate_spec('crx0', n_qubits=2)

    def test_parse_controlled_rotation_same_control_target(self):
        """Test that same control and target raises error."""
        from feynman_path.core.gates import parse_gate_spec

        with pytest.raises(ValueError, match="same control and target"):
            parse_gate_spec('crx0,0,pi/4', n_qubits=2)

    def test_parse_controlled_rotation_out_of_bounds(self):
        """Test that out-of-bounds qubits raise error."""
        from feynman_path.core.gates import parse_gate_spec

        # Control qubit out of bounds
        with pytest.raises(ValueError, match="out of range"):
            parse_gate_spec('crx5,1,pi', n_qubits=3)

        # Target qubit out of bounds
        with pytest.raises(ValueError, match="out of range"):
            parse_gate_spec('crx0,5,pi', n_qubits=3)


class TestStateApplicationCRx:
    """Test applying CRx gate to quantum states."""

    def test_crx_apply_to_00_state(self):
        """Test CRx|00⟩ = |00⟩ (control=0, no change)."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('crx0,1,pi/4', n_qubits=2)
        transitions = gate.apply_to_state('00')

        # Control is 0, so target should be unchanged
        assert len(transitions) == 1
        assert '00' in transitions
        assert abs(transitions['00'] - 1.0) < 1e-10

    def test_crx_apply_to_01_state(self):
        """Test CRx|01⟩ = |01⟩ (control=0, no change)."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('crx0,1,pi/3', n_qubits=2)
        transitions = gate.apply_to_state('01')

        # Control is 0, so target should be unchanged
        assert len(transitions) == 1
        assert '01' in transitions
        assert abs(transitions['01'] - 1.0) < 1e-10

    def test_crx_apply_to_10_state_pi_over_2(self):
        """Test CRx(π/2)|10⟩ creates superposition (control=1)."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('crx0,1,pi/2', n_qubits=2)
        transitions = gate.apply_to_state('10')

        # Control is 1, apply Rx(π/2) to target
        # Rx(π/2)|0⟩ = (1/√2)(|0⟩ - i|1⟩)
        # So CRx|10⟩ = (1/√2)(|10⟩ - i|11⟩)
        assert '10' in transitions
        assert '11' in transitions
        sqrt2_inv = 1 / np.sqrt(2)
        assert abs(transitions['10'] - sqrt2_inv) < 1e-10
        assert abs(transitions['11'] - (-1j * sqrt2_inv)) < 1e-10

    def test_crx_apply_to_11_state_pi(self):
        """Test CRx(π)|11⟩ flips target with phase."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('crx0,1,pi', n_qubits=2)
        transitions = gate.apply_to_state('11')

        # Rx(π)|1⟩ = -i|0⟩
        # So CRx(π)|11⟩ = -i|10⟩
        assert len(transitions) == 1
        assert '10' in transitions
        assert abs(transitions['10'] - (-1j)) < 1e-10

    def test_crx_in_multi_qubit_system(self):
        """Test CRx in 3+ qubit system."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('crx1,2,pi/2', n_qubits=3)

        # Apply to |010⟩ (control q1=1, target q2=0)
        transitions = gate.apply_to_state('010')

        # Should create superposition on q2
        assert '010' in transitions
        assert '011' in transitions


class TestStateApplicationCRy:
    """Test applying CRy gate to quantum states."""

    def test_cry_apply_to_00_state(self):
        """Test CRy|00⟩ = |00⟩ (control=0, no change)."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('cry0,1,pi/4', n_qubits=2)
        transitions = gate.apply_to_state('00')

        assert len(transitions) == 1
        assert '00' in transitions
        assert abs(transitions['00'] - 1.0) < 1e-10

    def test_cry_apply_to_10_state_pi_over_2(self):
        """Test CRy(π/2)|10⟩ creates equal superposition."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('cry0,1,pi/2', n_qubits=2)
        transitions = gate.apply_to_state('10')

        # Ry(π/2)|0⟩ = (1/√2)(|0⟩ + |1⟩)
        # So CRy|10⟩ = (1/√2)(|10⟩ + |11⟩)
        assert '10' in transitions
        assert '11' in transitions
        sqrt2_inv = 1 / np.sqrt(2)
        assert abs(transitions['10'] - sqrt2_inv) < 1e-10
        assert abs(transitions['11'] - sqrt2_inv) < 1e-10

    def test_cry_in_multi_qubit_system(self):
        """Test CRy in 3+ qubit system."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('cry1,2,pi/2', n_qubits=3)

        # Apply to |110⟩ (control q1=1, target q2=0)
        transitions = gate.apply_to_state('110')

        # Should create superposition on q2
        assert '110' in transitions
        assert '111' in transitions


class TestStateApplicationCRz:
    """Test applying CRz gate to quantum states."""

    def test_crz_apply_to_00_state(self):
        """Test CRz|00⟩ = |00⟩ (control=0, no change)."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('crz0,1,pi', n_qubits=2)
        transitions = gate.apply_to_state('00')

        assert len(transitions) == 1
        assert '00' in transitions
        assert abs(transitions['00'] - 1.0) < 1e-10

    def test_crz_apply_to_10_state_pi(self):
        """Test CRz(π)|10⟩ adds phase, no state change."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('crz0,1,pi', n_qubits=2)
        transitions = gate.apply_to_state('10')

        # Rz only adds phase, state stays |10⟩
        # Rz(π)|0⟩ = e^(-iπ/2)|0⟩ = -i|0⟩
        assert len(transitions) == 1
        assert '10' in transitions
        expected_phase = np.exp(-1j * np.pi / 2)
        assert abs(transitions['10'] - expected_phase) < 1e-10

    def test_crz_apply_to_11_state_pi_over_4(self):
        """Test CRz(π/4)|11⟩ adds phase."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('crz0,1,pi/4', n_qubits=2)
        transitions = gate.apply_to_state('11')

        # Rz(π/4)|1⟩ = e^(iπ/8)|1⟩
        assert len(transitions) == 1
        assert '11' in transitions
        expected_phase = np.exp(1j * np.pi / 8)
        assert abs(transitions['11'] - expected_phase) < 1e-10


class TestIntegrationControlledRotations:
    """Integration tests for controlled rotation gates."""

    def test_all_controlled_rotation_gates_different(self):
        """Test that CRx, CRy, CRz produce different matrices."""
        from feynman_path.core.gates import generate_crx_matrix, generate_cry_matrix, generate_crz_matrix

        theta = np.pi / 4
        CRx = generate_crx_matrix(theta)
        CRy = generate_cry_matrix(theta)
        CRz = generate_crz_matrix(theta)

        # All should be different
        assert not np.allclose(CRx, CRy)
        assert not np.allclose(CRx, CRz)
        assert not np.allclose(CRy, CRz)

    def test_controlled_rotation_composition(self):
        """Test CRx(θ₁)·CRx(θ₂) = CRx(θ₁+θ₂)."""
        from feynman_path.core.gates import generate_crx_matrix

        theta1 = np.pi / 4
        theta2 = np.pi / 6

        CRx1 = generate_crx_matrix(theta1)
        CRx2 = generate_crx_matrix(theta2)
        CRx_combined = generate_crx_matrix(theta1 + theta2)

        assert np.allclose(CRx1 @ CRx2, CRx_combined, atol=1e-10)

    def test_negative_angle(self):
        """Test CRx(-θ) = CRx(θ)† (inverse rotation)."""
        from feynman_path.core.gates import generate_crx_matrix

        theta = np.pi / 4
        CRx_neg = generate_crx_matrix(-theta)
        CRx_pos = generate_crx_matrix(theta)

        # CRx(-θ) should be the Hermitian conjugate of CRx(θ)
        assert np.allclose(CRx_neg, CRx_pos.conj().T, atol=1e-10)

    def test_crx_cry_crz_in_circuit(self):
        """Test parsing and applying multiple controlled rotations."""
        from feynman_path.core.gates import parse_gate_spec

        gates_specs = ['crx0,1,pi/4', 'cry1,2,pi/2', 'crz0,1,pi']
        n_qubits = 3

        for spec in gates_specs:
            gate = parse_gate_spec(spec, n_qubits=n_qubits)
            assert gate is not None
            assert hasattr(gate, 'angle')
            assert len(gate.qubits) == 2

    def test_controlled_rotation_with_cnot(self):
        """Test mixing controlled rotations with CNOT gates."""
        from feynman_path.core.gates import parse_gate_spec

        n_qubits = 2

        # Parse both types of gates
        crx_gate = parse_gate_spec('crx0,1,pi/4', n_qubits=n_qubits)
        cnot_gate = parse_gate_spec('cnot0,1', n_qubits=n_qubits)

        assert crx_gate.name == 'CRx'
        assert cnot_gate.name == 'CNOT'

    def test_feynman_graph_with_controlled_rotations(self):
        """Test full FeynmanGraph integration with controlled rotations."""
        from feynman_path.core import FeynmanGraph, parse_gate_spec

        graph = FeynmanGraph(n_qubits=2)

        # Apply H to qubit 0, then CRx
        h_gate = parse_gate_spec('h0', n_qubits=2)
        crx_gate = parse_gate_spec('crx0,1,pi/2', n_qubits=2)

        graph.apply_gate(h_gate)
        graph.apply_gate(crx_gate)

        # Should create entanglement
        final_state = graph.get_final_state()
        assert len(final_state) > 1  # Multiple states in superposition

    def test_controlled_rotation_json_output(self):
        """Test that controlled rotations work with JSON serialization."""
        from feynman_path.core import FeynmanGraph, parse_gate_spec

        graph = FeynmanGraph(n_qubits=2)

        crz_gate = parse_gate_spec('crz0,1,pi/4', n_qubits=2)
        graph.apply_gate(crz_gate)

        # Should be able to export to JSON
        json_data = graph.to_json()
        assert json_data is not None
        assert 'type' in json_data
        assert json_data['type'] == 'feynmanpath'
