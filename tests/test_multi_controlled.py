"""Tests for Toffoli and multi-controlled NOT gates."""

import pytest
import numpy as np


class TestToffoliGate:
    """Test Toffoli (CCX, CCNOT) gate."""

    def test_toffoli_matrix_definition(self):
        """Test Toffoli gate matrix definition."""
        from feynman_path.core.gates import GATE_MATRICES

        Toffoli = GATE_MATRICES['Toffoli']

        # Toffoli is 8x8 matrix (3 qubits)
        assert Toffoli.shape == (8, 8)

    def test_toffoli_is_unitary(self):
        """Test that Toffoli matrix is unitary."""
        from feynman_path.core.gates import GATE_MATRICES

        Toffoli = GATE_MATRICES['Toffoli']
        assert np.allclose(Toffoli @ Toffoli.conj().T, np.eye(8))

    def test_toffoli_is_self_inverse(self):
        """Test that Toffoli² = I (self-inverse)."""
        from feynman_path.core.gates import GATE_MATRICES

        Toffoli = GATE_MATRICES['Toffoli']
        assert np.allclose(Toffoli @ Toffoli, np.eye(8))

    def test_toffoli_matrix_structure(self):
        """Test Toffoli matrix has correct structure."""
        from feynman_path.core.gates import GATE_MATRICES

        Toffoli = GATE_MATRICES['Toffoli']

        # Expected: identity except swap rows 6 and 7 (|110⟩ ↔ |111⟩)
        # In basis ordering: |control0,control1,target⟩
        expected = np.eye(8, dtype=np.complex128)
        expected[6, 6] = 0
        expected[7, 7] = 0
        expected[6, 7] = 1
        expected[7, 6] = 1

        assert np.allclose(Toffoli, expected)

    def test_parse_toffoli_gate_spec(self):
        """Test parsing Toffoli gate specification."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('toffoli0,1,2', n_qubits=3)

        assert gate.name == 'Toffoli'
        assert gate.qubits == (0, 1, 2)
        assert gate.n_qubits == 3

    def test_toffoli_apply_both_controls_zero(self):
        """Test Toffoli |000⟩ → |000⟩ (both controls 0, no flip)."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('toffoli0,1,2', n_qubits=3)
        transitions = gate.apply_to_state('000')

        assert transitions == {'000': 1.0}

    def test_toffoli_apply_one_control_zero(self):
        """Test Toffoli |100⟩ → |100⟩ (one control 0, no flip)."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('toffoli0,1,2', n_qubits=3)
        transitions = gate.apply_to_state('100')

        assert transitions == {'100': 1.0}

    def test_toffoli_apply_both_controls_one_target_zero(self):
        """Test Toffoli |110⟩ → |111⟩ (both controls 1, flip target 0→1)."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('toffoli0,1,2', n_qubits=3)
        transitions = gate.apply_to_state('110')

        assert transitions == {'111': 1.0}

    def test_toffoli_apply_both_controls_one_target_one(self):
        """Test Toffoli |111⟩ → |110⟩ (both controls 1, flip target 1→0)."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('toffoli0,1,2', n_qubits=3)
        transitions = gate.apply_to_state('111')

        assert transitions == {'110': 1.0}

    def test_toffoli_all_eight_basis_states(self):
        """Test Toffoli on all 8 basis states."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('toffoli0,1,2', n_qubits=3)

        # Expected transitions
        expected = {
            '000': '000',  # No flip
            '001': '001',  # No flip
            '010': '010',  # No flip
            '011': '011',  # No flip
            '100': '100',  # No flip
            '101': '101',  # No flip
            '110': '111',  # Flip target
            '111': '110',  # Flip target
        }

        for input_state, output_state in expected.items():
            transitions = gate.apply_to_state(input_state)
            assert transitions == {output_state: 1.0}, \
                f"Toffoli {input_state} should give {output_state}"

    def test_toffoli_in_four_qubit_system(self):
        """Test Toffoli in a 4-qubit system."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('toffoli0,1,2', n_qubits=4)

        # Test |1100⟩ → |1110⟩ (q3 is spectator)
        transitions = gate.apply_to_state('1100')
        assert transitions == {'1110': 1.0}

        # Test |1101⟩ → |1111⟩
        transitions = gate.apply_to_state('1101')
        assert transitions == {'1111': 1.0}

    def test_toffoli_different_qubit_ordering(self):
        """Test Toffoli with different qubit indices."""
        from feynman_path.core.gates import parse_gate_spec

        # Toffoli on qubits 2,1,0
        gate = parse_gate_spec('toffoli2,1,0', n_qubits=3)

        # |110⟩ means q0=0, q1=1, q2=1
        # Controls q2=1, q1=1, so flip target q0: 0→1
        transitions = gate.apply_to_state('011')
        assert transitions == {'111': 1.0}

    def test_toffoli_non_consecutive_qubits(self):
        """Test Toffoli with non-consecutive qubit indices."""
        from feynman_path.core.gates import parse_gate_spec

        # Toffoli on qubits 0,2,4 in a 5-qubit system
        # control0=0, control1=2, target=4
        gate = parse_gate_spec('toffoli0,2,4', n_qubits=5)

        # |10100⟩: q0=1, q1=0, q2=1, q3=0, q4=0
        # Controls q0=1, q2=1, both 1, so flip target q4: 0→1
        transitions = gate.apply_to_state('10100')
        assert transitions == {'10101': 1.0}

        # |10101⟩: flip target q4: 1→0
        transitions = gate.apply_to_state('10101')
        assert transitions == {'10100': 1.0}

        # |10000⟩: q0=1, q2=0, not both controls 1, no flip
        transitions = gate.apply_to_state('10000')
        assert transitions == {'10000': 1.0}

    def test_toffoli_from_registry(self):
        """Test creating Toffoli gate from registry."""
        from feynman_path.core.gates import GATE_REGISTRY

        gate = GATE_REGISTRY['toffoli'](control0=0, control1=1, target=2, n_qubits=3)

        assert gate.name == 'Toffoli'
        assert gate.qubits == (0, 1, 2)


class TestMultiControlledNOT:
    """Test multi-controlled NOT (m[k]cnot) gates."""

    def test_m1cnot_behaves_like_cnot(self):
        """Test m1cnot (k=1) behaves like CNOT."""
        from feynman_path.core.gates import parse_gate_spec

        m1cnot = parse_gate_spec('m1cnot0,1', n_qubits=2)
        cnot = parse_gate_spec('cnot0,1', n_qubits=2)

        # Test on all basis states
        for i in range(4):
            state = format(i, '02b')[::-1]
            assert m1cnot.apply_to_state(state) == cnot.apply_to_state(state)

    def test_m2cnot_behaves_like_toffoli(self):
        """Test m2cnot (k=2) behaves like Toffoli."""
        from feynman_path.core.gates import parse_gate_spec

        m2cnot = parse_gate_spec('m2cnot0,1,2', n_qubits=3)
        toffoli = parse_gate_spec('toffoli0,1,2', n_qubits=3)

        # Test on all basis states
        for i in range(8):
            state = format(i, '03b')[::-1]
            assert m2cnot.apply_to_state(state) == toffoli.apply_to_state(state)

    def test_mcnot_is_self_inverse_via_application(self):
        """Test that m[k]cnot is self-inverse by applying twice."""
        from feynman_path.core.gates import parse_gate_spec

        for k in [1, 2, 3, 4]:
            n_qubits = k + 1
            qubit_str = ','.join(str(i) for i in range(n_qubits))
            gate = parse_gate_spec(f'm{k}cnot{qubit_str}', n_qubits=n_qubits)

            # Test on a few states
            for i in range(min(8, 2**n_qubits)):
                state = format(i, f'0{n_qubits}b')[::-1]
                # Apply twice
                first_result = gate.apply_to_state(state)
                assert len(first_result) == 1
                intermediate_state = list(first_result.keys())[0]
                second_result = gate.apply_to_state(intermediate_state)
                assert len(second_result) == 1
                final_state = list(second_result.keys())[0]
                # Should return to original
                assert final_state == state

    def test_parse_m2cnot_gate_spec(self):
        """Test parsing m2cnot (Toffoli) specification."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('m2cnot0,1,2', n_qubits=3)

        assert gate.name == 'M2CNOT'
        assert gate.qubits == (0, 1, 2)

    def test_parse_m3cnot_gate_spec(self):
        """Test parsing m3cnot specification."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('m3cnot0,1,2,3', n_qubits=4)

        assert gate.name == 'M3CNOT'
        assert gate.qubits == (0, 1, 2, 3)

    def test_parse_m5cnot_gate_spec(self):
        """Test parsing m5cnot specification."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('m5cnot0,1,2,3,4,5', n_qubits=6)

        assert gate.name == 'M5CNOT'
        assert gate.qubits == (0, 1, 2, 3, 4, 5)

    def test_m2cnot_apply_all_controls_one(self):
        """Test m2cnot with all controls = 1."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('m2cnot0,1,2', n_qubits=3)

        # |110⟩ → |111⟩ (both controls 1, flip target)
        transitions = gate.apply_to_state('110')
        assert transitions == {'111': 1.0}

    def test_m3cnot_apply_all_controls_one(self):
        """Test m3cnot with all controls = 1."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('m3cnot0,1,2,3', n_qubits=4)

        # |1110⟩ → |1111⟩ (all 3 controls = 1, flip target)
        transitions = gate.apply_to_state('1110')
        assert transitions == {'1111': 1.0}

        # |1111⟩ → |1110⟩ (all 3 controls = 1, flip target)
        transitions = gate.apply_to_state('1111')
        assert transitions == {'1110': 1.0}

    def test_m3cnot_apply_not_all_controls_one(self):
        """Test m3cnot with not all controls = 1."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('m3cnot0,1,2,3', n_qubits=4)

        # |0110⟩ → |0110⟩ (q0=0, not all controls 1, no flip)
        transitions = gate.apply_to_state('0110')
        assert transitions == {'0110': 1.0}

    def test_m3cnot_non_consecutive_qubits_case1(self):
        """Test m3cnot0,3,4,1 with non-consecutive qubits."""
        from feynman_path.core.gates import parse_gate_spec

        # m3cnot with controls on q0, q3, q4 and target on q1
        gate = parse_gate_spec('m3cnot0,3,4,1', n_qubits=5)

        # |10011⟩: q0=1, q1=0, q2=0, q3=1, q4=1
        # Controls q0=1, q3=1, q4=1, all 1, so flip target q1: 0→1
        transitions = gate.apply_to_state('10011')
        assert transitions == {'11011': 1.0}

        # |11011⟩: flip target q1: 1→0
        transitions = gate.apply_to_state('11011')
        assert transitions == {'10011': 1.0}

        # |10010⟩: q0=1, q3=1, q4=0, not all controls 1, no flip
        transitions = gate.apply_to_state('10010')
        assert transitions == {'10010': 1.0}

    def test_m3cnot_non_consecutive_qubits_case2(self):
        """Test m3cnot2,0,5,3 with scrambled qubit ordering."""
        from feynman_path.core.gates import parse_gate_spec

        # m3cnot with controls on q2, q0, q5 and target on q3
        gate = parse_gate_spec('m3cnot2,0,5,3', n_qubits=6)

        # |101001⟩: q0=1, q1=0, q2=1, q3=0, q4=0, q5=1
        # Controls q2=1, q0=1, q5=1, all 1, so flip target q3: 0→1
        transitions = gate.apply_to_state('101001')
        assert transitions == {'101101': 1.0}

        # |101101⟩: flip target q3: 1→0
        transitions = gate.apply_to_state('101101')
        assert transitions == {'101001': 1.0}

    def test_m3cnot_non_consecutive_spectator_qubits(self):
        """Test m3cnot with spectator qubits interspersed."""
        from feynman_path.core.gates import parse_gate_spec

        # m3cnot on q0,2,4,6 in 7-qubit system (q1, q3, q5 are spectators)
        gate = parse_gate_spec('m3cnot0,2,4,6', n_qubits=7)

        # |1010101⟩: q0=1, q1=0, q2=1, q3=0, q4=1, q5=0, q6=1
        # Controls q0=1, q2=1, q4=1, all 1
        # Target q6=1, flip to 0
        # Spectators q1=0, q3=0, q5=0 unchanged
        transitions = gate.apply_to_state('1010101')
        assert transitions == {'1010100': 1.0}

        # With different spectator values
        # |1110111⟩: q0=1, q1=1, q2=1, q3=1, q4=1, q5=1, q6=1
        # Same controls all 1, flip target
        transitions = gate.apply_to_state('1110111')
        assert transitions == {'1110110': 1.0}

    def test_mcnot_k_out_of_bounds_too_small(self):
        """Test that k < 1 raises error."""
        from feynman_path.core.gates import parse_gate_spec

        with pytest.raises(ValueError, match="out of bounds"):
            parse_gate_spec('m0cnot0', n_qubits=2)

    def test_mcnot_k_out_of_bounds_too_large(self):
        """Test that k > 15 raises error."""
        from feynman_path.core.gates import parse_gate_spec

        # Try to create m16cnot (16 controls)
        qubit_indices = ','.join(str(i) for i in range(17))  # 17 qubits total
        with pytest.raises(ValueError, match="out of bounds"):
            parse_gate_spec(f'm16cnot{qubit_indices}', n_qubits=17)

    def test_m3cnot_correct_behavior(self):
        """Test m3cnot behaves correctly as 3-controlled NOT."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('m3cnot0,1,2,3', n_qubits=4)

        # Only when all 3 controls (q0, q1, q2) are 1 should target flip
        # |1110⟩ → |1111⟩ (all controls 1, target 0→1)
        assert gate.apply_to_state('1110') == {'1111': 1.0}

        # |1111⟩ → |1110⟩ (all controls 1, target 1→0)
        assert gate.apply_to_state('1111') == {'1110': 1.0}

        # |0110⟩ → |0110⟩ (q0=0, not all controls 1, no flip)
        assert gate.apply_to_state('0110') == {'0110': 1.0}

    def test_m1cnot_apply_to_states(self):
        """Test m1cnot (CNOT) application."""
        from feynman_path.core.gates import parse_gate_spec

        gate = parse_gate_spec('m1cnot0,1', n_qubits=2)

        # m1cnot should behave like CNOT
        assert gate.apply_to_state('10') == {'11': 1.0}
        assert gate.apply_to_state('11') == {'10': 1.0}
        assert gate.apply_to_state('00') == {'00': 1.0}
        assert gate.apply_to_state('01') == {'01': 1.0}

    def test_mcnot_in_larger_system(self):
        """Test m[k]cnot in larger qubit system."""
        from feynman_path.core.gates import parse_gate_spec

        # m2cnot on qubits 1,2,3 in a 5-qubit system
        gate = parse_gate_spec('m2cnot1,2,3', n_qubits=5)

        # |01100⟩: q0=0, q1=1, q2=1, q3=0, q4=0
        # Controls q1=1, q2=1, so flip q3: 0→1
        transitions = gate.apply_to_state('01100')
        assert transitions == {'01110': 1.0}


class TestMultiControlledIntegration:
    """Integration tests for multi-controlled gates."""

    def test_toffoli_equals_m2cnot(self):
        """Test that toffoli and m2cnot give same results."""
        from feynman_path.core.gates import parse_gate_spec

        toffoli = parse_gate_spec('toffoli0,1,2', n_qubits=3)
        m2cnot = parse_gate_spec('m2cnot0,1,2', n_qubits=3)

        # Test on all basis states
        for i in range(8):
            state = format(i, '03b')[::-1]  # Convert to bitstring
            toffoli_result = toffoli.apply_to_state(state)
            m2cnot_result = m2cnot.apply_to_state(state)
            assert toffoli_result == m2cnot_result

    def test_all_mcnot_gates_available(self):
        """Test that m[k]cnot gates can be created for k=1 to 15."""
        from feynman_path.core.gates import parse_gate_spec

        for k in range(1, 16):
            n_qubits = k + 1
            qubit_str = ','.join(str(i) for i in range(n_qubits))
            gate = parse_gate_spec(f'm{k}cnot{qubit_str}', n_qubits=n_qubits)
            assert gate.name == f'M{k}CNOT'
            assert len(gate.qubits) == n_qubits

    def test_mcnot_general_property(self):
        """Test general property: flips target only when all controls = 1."""
        from feynman_path.core.gates import parse_gate_spec

        for k in [2, 3, 4]:
            n_qubits = k + 1
            qubit_str = ','.join(str(i) for i in range(n_qubits))
            gate = parse_gate_spec(f'm{k}cnot{qubit_str}', n_qubits=n_qubits)

            # State with all controls = 1, target = 0
            # Bitstring: first k bits are 1, last bit is 0
            all_ones_target_zero = '1' * k + '0'
            all_ones_target_one = '1' * k + '1'

            # Should flip target
            result_0 = gate.apply_to_state(all_ones_target_zero)
            assert result_0 == {all_ones_target_one: 1.0}

            result_1 = gate.apply_to_state(all_ones_target_one)
            assert result_1 == {all_ones_target_zero: 1.0}

    def test_m4cnot_non_consecutive_comprehensive(self):
        """Test m4cnot with various non-consecutive qubit patterns."""
        from feynman_path.core.gates import parse_gate_spec

        # m4cnot on q0,2,5,7,3 (4 controls + 1 target, 8-qubit system)
        gate = parse_gate_spec('m4cnot0,2,5,7,3', n_qubits=8)

        # |10100101⟩: q0=1, q1=0, q2=1, q3=0, q4=0, q5=1, q6=0, q7=1
        # Controls q0=1, q2=1, q5=1, q7=1, all 1
        # Target q3=0, flip to 1
        transitions = gate.apply_to_state('10100101')
        assert transitions == {'10110101': 1.0}

        # Not all controls 1
        # |10100001⟩: q5=0, so no flip
        transitions = gate.apply_to_state('10100001')
        assert transitions == {'10100001': 1.0}


class TestRepeatedQubitValidation:
    """Test that multi-qubit gates reject repeated qubits."""

    def test_cnot_same_control_and_target(self):
        """Test that CNOT rejects same control and target qubit."""
        from feynman_path.core.gates import parse_gate_spec

        with pytest.raises(ValueError, match="cannot have same control and target"):
            parse_gate_spec('cnot1,1', n_qubits=3)

    def test_toffoli_repeated_qubits(self):
        """Test that Toffoli rejects repeated qubits."""
        from feynman_path.core.gates import parse_gate_spec

        with pytest.raises(ValueError, match="cannot have repeated qubits"):
            parse_gate_spec('toffoli1,1,2', n_qubits=3)

        with pytest.raises(ValueError, match="cannot have repeated qubits"):
            parse_gate_spec('toffoli0,1,1', n_qubits=3)

    def test_m3cnot_repeated_qubits(self):
        """Test that m3cnot rejects repeated qubits."""
        from feynman_path.core.gates import parse_gate_spec

        # Repeated in controls
        with pytest.raises(ValueError, match="cannot have repeated qubits"):
            parse_gate_spec('m3cnot0,1,1,3', n_qubits=4)

        # Target same as control
        with pytest.raises(ValueError, match="cannot have repeated qubits"):
            parse_gate_spec('m3cnot0,1,2,1', n_qubits=4)
