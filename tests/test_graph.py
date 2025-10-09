"""Tests for FeynmanGraph class and Feynman path computation."""

import pytest
import sympy
import json


class TestFeynmanGraphBasics:
    """Test basic FeynmanGraph functionality."""

    def test_graph_initialization(self):
        """Test creating a FeynmanGraph."""
        from feynman_path.core.graph import FeynmanGraph

        graph = FeynmanGraph(n_qubits=2)

        assert graph.n_qubits == 2
        assert graph.n_timesteps == 0  # No gates applied yet

    def test_graph_with_custom_initial_state(self):
        """Test FeynmanGraph with non-|00...0⟩ initial state."""
        from feynman_path.core.graph import FeynmanGraph

        # Start with |11⟩ instead of |00⟩
        graph = FeynmanGraph(n_qubits=2, initial_state='11')

        assert graph.n_qubits == 2
        # Initial state should be set

    def test_apply_single_gate(self):
        """Test applying a single gate to the graph."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec

        graph = FeynmanGraph(n_qubits=1)
        gate = parse_gate_spec('h0', n_qubits=1)

        graph.apply_gate(gate)

        assert graph.n_timesteps == 1

    def test_apply_multiple_gates(self):
        """Test applying multiple gates in sequence."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec

        graph = FeynmanGraph(n_qubits=2)

        h0 = parse_gate_spec('h0', n_qubits=2)
        cnot = parse_gate_spec('cnot0,1', n_qubits=2)

        graph.apply_gate(h0)
        graph.apply_gate(cnot)

        assert graph.n_timesteps == 2


class TestFeynmanGraphJSONStructure:
    """Test JSON output structure and format."""

    def test_json_output_basic_structure(self):
        """Test that JSON has correct basic structure."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec

        graph = FeynmanGraph(n_qubits=2)
        h0 = parse_gate_spec('h0', n_qubits=2)
        graph.apply_gate(h0)

        json_data = graph.to_json()

        assert 'type' in json_data
        assert json_data['type'] == 'feynmanpath'
        assert 'cols' in json_data
        assert isinstance(json_data['cols'], list)

    def test_json_cols_structure(self):
        """Test that each col has correct structure."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec

        graph = FeynmanGraph(n_qubits=2)
        h0 = parse_gate_spec('h0', n_qubits=2)
        graph.apply_gate(h0)

        json_data = graph.to_json()
        cols = json_data['cols']

        # Should have at least one column
        assert len(cols) > 0

        # Each column is a dict with state keys
        for col in cols:
            assert isinstance(col, dict)
            for state_key, state_data in col.items():
                assert isinstance(state_key, str)  # State like '00'
                assert 'amp' in state_data
                assert 'next' in state_data
                assert isinstance(state_data['next'], dict)

    def test_hadamard_json_output(self):
        """Test JSON output for single Hadamard gate."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec

        graph = FeynmanGraph(n_qubits=1)
        h0 = parse_gate_spec('h0', n_qubits=1)
        graph.apply_gate(h0)

        json_data = graph.to_json()

        # Should have one column (one gate)
        assert len(json_data['cols']) == 1

        col0 = json_data['cols'][0]
        # Initial state |0⟩
        assert '0' in col0
        assert col0['0']['amp'] == 1

        # Transitions to |0⟩ and |1⟩
        assert '0' in col0['0']['next']
        assert '1' in col0['0']['next']

    def test_two_qubit_hadamard_json(self):
        """Test JSON for H gate on 2-qubit system."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec

        graph = FeynmanGraph(n_qubits=2)
        h0 = parse_gate_spec('h0', n_qubits=2)
        graph.apply_gate(h0)

        json_data = graph.to_json()
        col0 = json_data['cols'][0]

        # Initial state |00⟩
        assert '00' in col0
        assert col0['00']['amp'] == 1

        # Transitions
        next_states = col0['00']['next']
        assert '00' in next_states
        assert '10' in next_states  # q0 flips


class TestFeynmanGraphAmplitudes:
    """Test that amplitudes are computed correctly."""

    def test_hadamard_amplitudes(self):
        """Test H gate produces correct amplitudes."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec

        graph = FeynmanGraph(n_qubits=1)
        h0 = parse_gate_spec('h0', n_qubits=1)
        graph.apply_gate(h0)

        json_data = graph.to_json()
        transitions = json_data['cols'][0]['0']['next']

        # H|0⟩ = (|0⟩ + |1⟩)/√2
        # Check if amplitudes are sqrt(2)/2 (as sympy expressions or floats)
        amp_0 = transitions['0']
        amp_1 = transitions['1']

        # Convert to sympy if needed
        if not isinstance(amp_0, sympy.Basic):
            amp_0 = sympy.sympify(str(amp_0))
        if not isinstance(amp_1, sympy.Basic):
            amp_1 = sympy.sympify(str(amp_1))

        expected = sympy.sqrt(2) / 2
        assert amp_0 == expected or abs(float(amp_0) - float(expected)) < 1e-10
        assert amp_1 == expected or abs(float(amp_1) - float(expected)) < 1e-10

    def test_z_gate_phase(self):
        """Test Z gate produces correct phase."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec

        # Start with |1⟩
        graph = FeynmanGraph(n_qubits=1, initial_state='1')
        z0 = parse_gate_spec('z0', n_qubits=1)
        graph.apply_gate(z0)

        json_data = graph.to_json()
        col0 = json_data['cols'][0]

        # Z|1⟩ = -|1⟩
        assert '1' in col0
        assert col0['1']['next']['1'] == -1 or str(col0['1']['next']['1']) == '-1'


class TestComplexCircuits:
    """Test more complex quantum circuits."""

    def test_bell_pair_creation(self):
        """Test creating a Bell pair: H on q0, then CNOT(0,1)."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec

        graph = FeynmanGraph(n_qubits=2)

        # H on qubit 0
        h0 = parse_gate_spec('h0', n_qubits=2)
        graph.apply_gate(h0)

        # CNOT(0, 1)
        cnot = parse_gate_spec('cnot0,1', n_qubits=2)
        graph.apply_gate(cnot)

        json_data = graph.to_json()

        # Should have 2 columns
        assert len(json_data['cols']) == 2

        # After these gates, final states should be |00⟩ and |11⟩ (Bell pair)
        # Check last column has the right structure

    def test_three_gate_sequence(self):
        """Test H, CNOT, Z sequence."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec

        graph = FeynmanGraph(n_qubits=2)

        gates = ['h0', 'cnot0,1', 'z1']
        for gate_str in gates:
            gate = parse_gate_spec(gate_str, n_qubits=2)
            graph.apply_gate(gate)

        json_data = graph.to_json()

        # Should have 3 columns
        assert len(json_data['cols']) == 3

    def test_interference_circuit_structure(self):
        """Test the interference circuit from the example."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec

        graph = FeynmanGraph(n_qubits=2)

        # The interference circuit
        gate_sequence = ['h0', 'cnot0,1', 'z1', 'h0', 'h1', 'cnot1,0', 'h1']

        for gate_str in gate_sequence:
            gate = parse_gate_spec(gate_str, n_qubits=2)
            graph.apply_gate(gate)

        json_data = graph.to_json()

        # Should have 7 columns (7 gates)
        assert len(json_data['cols']) == 7

        # Verify structure is maintained throughout
        for col in json_data['cols']:
            for state_key, state_data in col.items():
                assert 'amp' in state_data
                assert 'next' in state_data


class TestJSONSerialization:
    """Test JSON serialization with sympy expressions."""

    def test_sympy_expression_in_json(self):
        """Test that sympy expressions are properly serialized."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec

        graph = FeynmanGraph(n_qubits=1)
        h0 = parse_gate_spec('h0', n_qubits=1)
        graph.apply_gate(h0)

        json_data = graph.to_json()

        # Convert to JSON string and back to verify serializability
        from feynman_path.core.serialization import to_json_string

        json_str = to_json_string(json_data)
        assert isinstance(json_str, str)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed['type'] == 'feynmanpath'

    def test_complex_amplitudes_serialization(self):
        """Test serialization of complex amplitudes."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec

        graph = FeynmanGraph(n_qubits=1, initial_state='1')
        s0 = parse_gate_spec('s0', n_qubits=1)  # S gate adds phase i
        graph.apply_gate(s0)

        json_data = graph.to_json()

        # Should be serializable
        from feynman_path.core.serialization import to_json_string
        json_str = to_json_string(json_data)
        assert isinstance(json_str, str)


class TestLargeCircuits:
    """Test scalability with larger circuits."""

    def test_5_qubit_circuit(self):
        """Test 5-qubit circuit with multiple gates."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec
        import numpy as np

        # Create the 5-qubit circuit
        graph = FeynmanGraph(n_qubits=5)
        gates = ['h0', 'h1', 'h2', 'h3', 'h4', 'cnot0,1', 'z1',
                 'cnot1,2', 'cnot2,3', 'cnot1,0', 'h0', 'h1', 'h2', 'h3', 'h4']

        for gate_str in gates:
            gate = parse_gate_spec(gate_str, n_qubits=5)
            graph.apply_gate(gate)

        json_data = graph.to_json()

        # Should have 15 columns (15 gates)
        assert len(json_data['cols']) == 15

        # Verify structure is maintained
        for col in json_data['cols']:
            assert isinstance(col, dict)
            for state_key, state_data in col.items():
                assert 'amp' in state_data
                assert 'next' in state_data
                assert isinstance(state_data['next'], dict)

        # Check first column: H on qubit 0
        s2 = np.sqrt(2)
        col0 = json_data['cols'][0]
        assert '00000' in col0
        assert abs(col0['00000']['amp'] - 1.0) < 1e-10
        assert abs(col0['00000']['next']['00000'] - s2/2) < 1e-10
        assert abs(col0['00000']['next']['10000'] - s2/2) < 1e-10

        # Check that column 5 (after 5 Hadamards) has 32 states
        col5 = json_data['cols'][5]
        assert len(col5) == 32  # All 2^5 states

        # All states in column 5 should have amplitude sqrt(2)/8
        expected_amp = s2 / 8
        for state_data in col5.values():
            assert abs(abs(state_data['amp']) - expected_amp) < 1e-10

        # Final state should be |01000⟩
        final_state = graph.get_final_state()
        assert len(final_state) == 1
        assert '01000' in final_state
        assert abs(final_state['01000'] - 1.0) < 1e-8
