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
        # Should have 2 columns: one for gate, one final layer
        assert len(json_data['cols']) == 2

    def test_json_cols_structure(self):
        """Test that each col has correct structure."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec

        graph = FeynmanGraph(n_qubits=2)
        h0 = parse_gate_spec('h0', n_qubits=2)
        graph.apply_gate(h0)

        json_data = graph.to_json()
        cols = json_data['cols']

        # Should have 2 columns (gate + final layer)
        assert len(cols) == 2

        # Each column is a dict with state keys
        for col in cols:
            assert isinstance(col, dict)
            for state_key, state_data in col.items():
                assert isinstance(state_key, str)  # State like '00'
                assert 'amp' in state_data
                assert 'next' in state_data
                assert isinstance(state_data['next'], dict)

        # Final column should have empty 'next' for all states
        final_col = cols[-1]
        for state_data in final_col.values():
            assert state_data['next'] == {}

    def test_hadamard_json_output(self):
        """Test JSON output for single Hadamard gate."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec

        graph = FeynmanGraph(n_qubits=1)
        h0 = parse_gate_spec('h0', n_qubits=1)
        graph.apply_gate(h0)

        json_data = graph.to_json()

        # Should have 2 columns (one gate + final layer)
        assert len(json_data['cols']) == 2

        col0 = json_data['cols'][0]
        # Initial state |0⟩
        assert '0' in col0
        assert col0['0']['amp'] == 1

        # Transitions to |0⟩ and |1⟩ with cumulative amplitudes
        assert '0' in col0['0']['next']
        assert '1' in col0['0']['next']

        # Final layer should have both output states with empty next
        final_col = json_data['cols'][1]
        assert '0' in final_col
        assert '1' in final_col
        assert final_col['0']['next'] == {}
        assert final_col['1']['next'] == {}

    def test_two_qubit_hadamard_json(self):
        """Test JSON for H gate on 2-qubit system."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec

        graph = FeynmanGraph(n_qubits=2)
        h0 = parse_gate_spec('h0', n_qubits=2)
        graph.apply_gate(h0)

        json_data = graph.to_json()

        # Should have 2 columns (gate + final layer)
        assert len(json_data['cols']) == 2

        col0 = json_data['cols'][0]

        # Initial state |00⟩
        assert '00' in col0
        assert col0['00']['amp'] == 1

        # Transitions
        next_states = col0['00']['next']
        assert '00' in next_states
        assert '10' in next_states  # q0 flips

        # Final layer check
        final_col = json_data['cols'][1]
        assert '00' in final_col
        assert '10' in final_col
        assert final_col['00']['next'] == {}
        assert final_col['10']['next'] == {}


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

        # Should have 3 columns (H, CNOT, final layer)
        assert len(json_data['cols']) == 3

        # After these gates, final states should be |00⟩ and |11⟩ (Bell pair)
        # Check last column has the right structure
        final_col = json_data['cols'][2]
        assert '00' in final_col
        assert '11' in final_col
        assert final_col['00']['next'] == {}
        assert final_col['11']['next'] == {}

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

        # Should have 4 columns (3 gates + final layer)
        assert len(json_data['cols']) == 4

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

        # Should have 8 columns (7 gates + final layer)
        assert len(json_data['cols']) == 8

        # Verify structure is maintained throughout
        for col in json_data['cols']:
            for state_key, state_data in col.items():
                assert 'amp' in state_data
                assert 'next' in state_data

        # Final column should have empty 'next'
        final_col = json_data['cols'][-1]
        for state_data in final_col.values():
            assert state_data['next'] == {}


class TestCumulativeAmplitudes:
    """Test that 'next' stores cumulative amplitudes (amp * transition_amplitude)."""

    def test_next_stores_cumulative_amplitudes_with_custom_initial(self):
        """Test that next stores cumulative amplitudes, not just transition amplitudes."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec
        import numpy as np

        # Create graph and manually set initial state to have amplitude 0.5
        graph = FeynmanGraph(n_qubits=2)
        graph.current_state.clear()
        graph.current_state.set('00', amplitude=0.5)

        # Apply H on qubit 0
        h0 = parse_gate_spec('h0', n_qubits=2)
        graph.apply_gate(h0)

        json_data = graph.to_json()
        col0 = json_data['cols'][0]

        # State '00' has amp=0.5
        assert col0['00']['amp'] == 0.5

        # next should contain cumulative: 0.5 * sqrt(2)/2 ≈ 0.3536
        expected_cumulative = 0.5 * np.sqrt(2) / 2
        assert abs(col0['00']['next']['00'] - expected_cumulative) < 1e-10
        assert abs(col0['00']['next']['10'] - expected_cumulative) < 1e-10

    def test_bell_pair_cumulative_amplitudes(self):
        """Test Bell pair creation has cumulative amplitudes in next."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec
        import numpy as np

        graph = FeynmanGraph(n_qubits=2)

        # Apply H on qubit 0
        h0 = parse_gate_spec('h0', n_qubits=2)
        graph.apply_gate(h0)

        # Apply CNOT(0,1)
        cnot = parse_gate_spec('cnot0,1', n_qubits=2)
        graph.apply_gate(cnot)

        json_data = graph.to_json()

        # Column 0 (after H): '00' has amp=1, next should be 1 * sqrt(2)/2
        col0 = json_data['cols'][0]
        s2_2 = np.sqrt(2) / 2
        assert abs(col0['00']['next']['00'] - s2_2) < 1e-10
        assert abs(col0['00']['next']['10'] - s2_2) < 1e-10

        # Column 1 (after CNOT): transitions are identity (1), but cumulative should be sqrt(2)/2
        col1 = json_data['cols'][1]
        # State '00' has amp=sqrt(2)/2, CNOT doesn't change it, so next should be sqrt(2)/2 * 1
        assert '00' in col1
        assert abs(col1['00']['amp'] - s2_2) < 1e-10
        assert abs(col1['00']['next']['00'] - s2_2) < 1e-10

        # State '10' has amp=sqrt(2)/2, CNOT flips to '11', so next should be sqrt(2)/2 * 1
        assert '10' in col1
        assert abs(col1['10']['amp'] - s2_2) < 1e-10
        assert abs(col1['10']['next']['11'] - s2_2) < 1e-10

    def test_multi_gate_cumulative_amplitudes(self):
        """Test cumulative amplitudes through multiple gates."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec
        import numpy as np

        graph = FeynmanGraph(n_qubits=1)

        # H gate creates superposition
        h0 = parse_gate_spec('h0', n_qubits=1)
        graph.apply_gate(h0)

        # Another H gate should collapse back
        graph.apply_gate(h0)

        json_data = graph.to_json()
        s2_2 = np.sqrt(2) / 2

        # Column 1 (second H): both states '0' and '1' have amp=sqrt(2)/2
        col1 = json_data['cols'][1]
        # State '0' with amp=s2_2 transitions to '0' and '1', cumulative should be s2_2 * s2_2 = 0.5
        assert abs(col1['0']['next']['0'] - 0.5) < 1e-10
        assert abs(col1['0']['next']['1'] - 0.5) < 1e-10


class TestFinalLayer:
    """Test that JSON output includes final layer with empty 'next'."""

    def test_json_includes_final_layer_single_gate(self):
        """Test single gate circuit has final layer."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec

        graph = FeynmanGraph(n_qubits=1)
        h0 = parse_gate_spec('h0', n_qubits=1)
        graph.apply_gate(h0)

        json_data = graph.to_json()

        # Should have 2 columns: one for gate, one final layer
        assert len(json_data['cols']) == 2

        # Column 1 (final layer)
        final_col = json_data['cols'][1]
        assert '0' in final_col
        assert '1' in final_col

        # Both states should have empty next
        assert final_col['0']['next'] == {}
        assert final_col['1']['next'] == {}

        # Final amplitudes should be sqrt(2)/2
        import numpy as np
        s2_2 = np.sqrt(2) / 2
        assert abs(final_col['0']['amp'] - s2_2) < 1e-10
        assert abs(final_col['1']['amp'] - s2_2) < 1e-10

    def test_json_includes_final_layer_bell_pair(self):
        """Test Bell pair circuit has final layer."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec

        graph = FeynmanGraph(n_qubits=2)
        h0 = parse_gate_spec('h0', n_qubits=2)
        cnot = parse_gate_spec('cnot0,1', n_qubits=2)

        graph.apply_gate(h0)
        graph.apply_gate(cnot)

        json_data = graph.to_json()

        # Should have 3 columns: H, CNOT, final
        assert len(json_data['cols']) == 3

        # Column 2 (final layer)
        final_col = json_data['cols'][2]
        assert '00' in final_col
        assert '11' in final_col

        # Both states should have empty next
        assert final_col['00']['next'] == {}
        assert final_col['11']['next'] == {}

    def test_final_layer_matches_final_state(self):
        """Test final layer amplitudes match get_final_state()."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec

        graph = FeynmanGraph(n_qubits=2)
        gates = ['h0', 'cnot0,1', 'z1']

        for gate_str in gates:
            gate = parse_gate_spec(gate_str, n_qubits=2)
            graph.apply_gate(gate)

        json_data = graph.to_json()
        final_state = graph.get_final_state()

        # Last column should be the final layer
        final_col = json_data['cols'][-1]

        # States in final column should match final_state
        assert set(final_col.keys()) == set(final_state.keys())

        # Amplitudes should match
        for state in final_col:
            assert final_col[state]['amp'] == final_state[state]
            assert final_col[state]['next'] == {}

    def test_json_empty_circuit_has_initial_layer(self):
        """Test empty circuit (no gates) has one layer with initial state."""
        from feynman_path.core.graph import FeynmanGraph

        graph = FeynmanGraph(n_qubits=2)
        # Don't apply any gates

        json_data = graph.to_json()

        # Should have 1 column (just the initial/final state)
        assert len(json_data['cols']) == 1

        # Column 0 should have only '00' with amp=1 and empty next
        col0 = json_data['cols'][0]
        assert len(col0) == 1
        assert '00' in col0
        assert col0['00']['amp'] == 1
        assert col0['00']['next'] == {}


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

        # Convert to JSON-like string with unquoted structural keys
        from feynman_path.core.serialization import to_json_string

        json_str = to_json_string(json_data)
        assert isinstance(json_str, str)

        # Should contain unquoted structural keys
        assert 'type: "feynmanpath"' in json_str
        assert 'cols:' in json_str
        assert 'amp:' in json_str
        assert 'next:' in json_str

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

        # Should have 16 columns (15 gates + final layer)
        assert len(json_data['cols']) == 16

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
        # Cumulative amplitudes: 1.0 * sqrt(2)/2
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

        # Final column should have empty next
        final_col = json_data['cols'][-1]
        assert '01000' in final_col
        assert final_col['01000']['next'] == {}
