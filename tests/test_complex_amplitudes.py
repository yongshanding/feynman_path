"""Tests for circuits with complex amplitudes."""

import pytest
import numpy as np


class TestComplexAmplitudeCircuits:
    """Test circuits that produce complex amplitudes."""

    def test_s_gate_produces_complex_amplitude(self):
        """Test that S gate produces imaginary amplitude."""
        from feynman_path.core import FeynmanGraph, parse_gate_spec

        # S gate on |1⟩ should give i|1⟩
        graph = FeynmanGraph(n_qubits=1, initial_state='1')
        s0 = parse_gate_spec('s0', n_qubits=1)
        graph.apply_gate(s0)

        json_data = graph.to_json()
        col0 = json_data['cols'][0]

        # Check that the amplitude is purely imaginary
        amp = col0['1']['next']['1']
        assert isinstance(amp, complex)
        assert abs(amp - 1j) < 1e-10

    def test_circuit_with_s_and_hadamard_gates(self):
        """Test circuit: h0 cnot0,1 s1 h0 h1 cnot1,0 h1"""
        from feynman_path.core import FeynmanGraph, parse_gate_spec

        graph = FeynmanGraph(n_qubits=2)
        gates = ['h0', 'cnot0,1', 's1', 'h0', 'h1', 'cnot1,0', 'h1']

        for gate_str in gates:
            gate = parse_gate_spec(gate_str, n_qubits=2)
            graph.apply_gate(gate)

        json_data = graph.to_json()
        s2 = np.sqrt(2)

        # Should have 7 columns
        assert len(json_data['cols']) == 7

        # Column 0: H on q0
        col0 = json_data['cols'][0]
        assert '00' in col0
        assert abs(col0['00']['amp'] - 1.0) < 1e-10
        assert abs(col0['00']['next']['00'] - s2/2) < 1e-10
        assert abs(col0['00']['next']['10'] - s2/2) < 1e-10

        # Column 1: CNOT(0,1)
        col1 = json_data['cols'][1]
        assert '00' in col1
        assert '10' in col1
        assert abs(col1['00']['amp'] - s2/2) < 1e-10
        assert abs(col1['10']['amp'] - s2/2) < 1e-10

        # Column 2: S on q1 - this creates complex amplitudes
        col2 = json_data['cols'][2]
        assert '00' in col2
        assert '11' in col2
        # State 00 (q1=0) unchanged
        assert abs(col2['00']['amp'] - s2/2) < 1e-10
        assert abs(col2['00']['next']['00'] - 1.0) < 1e-10
        # State 11 (q1=1) gets phase i
        assert abs(col2['11']['amp'] - s2/2) < 1e-10
        # S|1⟩ = i|1⟩, so amplitude becomes i*sqrt(2)/2
        assert abs(col2['11']['next']['11'] - 1j) < 1e-10

        # Column 3: H on q0 - complex amplitudes propagate
        col3 = json_data['cols'][3]
        assert '00' in col3
        assert '11' in col3
        assert abs(col3['00']['amp'] - s2/2) < 1e-10
        # State 11 now has complex amplitude
        assert isinstance(col3['11']['amp'], complex)
        assert abs(col3['11']['amp'] - s2/2*1j) < 1e-10

        # Column 4: H on q1
        col4 = json_data['cols'][4]
        assert len(col4) == 4  # Should have 4 states now
        # Check that we have complex amplitudes
        assert abs(col4['00']['amp'] - 0.5) < 1e-10
        assert abs(col4['01']['amp'] - 0.5*1j) < 1e-10
        assert abs(col4['10']['amp'] - 0.5) < 1e-10
        assert abs(col4['11']['amp'] - (-0.5*1j)) < 1e-10

        # Column 5: CNOT(1,0)
        col5 = json_data['cols'][5]
        assert len(col5) == 4
        # After CNOT, amplitudes should be sqrt(2)/4 ± 0.25*sqrt(2)*I
        expected_amp_1 = s2/4 + 0.25*s2*1j
        expected_amp_2 = s2/4 - 0.25*s2*1j

        assert abs(col5['00']['amp'] - expected_amp_1) < 1e-10
        assert abs(col5['01']['amp'] - expected_amp_2) < 1e-10
        assert abs(col5['10']['amp'] - expected_amp_2) < 1e-10
        assert abs(col5['11']['amp'] - expected_amp_1) < 1e-10

        # Column 6: Final H on q1
        col6 = json_data['cols'][6]
        assert len(col6) == 4

        # Check final state
        final_state = graph.get_final_state()
        assert len(final_state) == 2
        assert '00' in final_state
        assert '10' in final_state

        # Final amplitudes should be approximately 0.5 ± 0.5j
        assert abs(final_state['00'] - (0.5 + 0.5j)) < 1e-8
        assert abs(final_state['10'] - (0.5 - 0.5j)) < 1e-8

    def test_complex_amplitude_serialization(self):
        """Test that complex amplitudes are properly serialized to JSON."""
        from feynman_path.core import FeynmanGraph, parse_gate_spec, to_json_string

        graph = FeynmanGraph(n_qubits=1, initial_state='1')
        s0 = parse_gate_spec('s0', n_qubits=1)
        graph.apply_gate(s0)

        json_data = graph.to_json()
        json_str = to_json_string(json_data)

        # Should contain complex number representation
        assert '1j' in json_str or 'I' in json_str  # Python or sympy notation
        assert isinstance(json_str, str)

    def test_final_probabilities_with_complex_amplitudes(self):
        """Test that probabilities sum to 1 even with complex amplitudes."""
        from feynman_path.core import FeynmanGraph, parse_gate_spec

        graph = FeynmanGraph(n_qubits=2)
        gates = ['h0', 'cnot0,1', 's1', 'h0', 'h1', 'cnot1,0', 'h1']

        for gate_str in gates:
            gate = parse_gate_spec(gate_str, n_qubits=2)
            graph.apply_gate(gate)

        final_state = graph.get_final_state()

        # Calculate total probability
        total_prob = sum(abs(amp)**2 for amp in final_state.values())

        # Should sum to 1 (within numerical error)
        assert abs(total_prob - 1.0) < 1e-10
