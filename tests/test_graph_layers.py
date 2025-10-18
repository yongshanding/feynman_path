"""Tests for FeynmanGraph layer application functionality."""

import pytest
import numpy as np


class TestFeynmanGraphLayerApplication:
    """Test applying gates as layers in FeynmanGraph."""

    def test_apply_layer_single_gate(self):
        """Test that applying a layer with single gate behaves like apply_gate()."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec

        graph = FeynmanGraph(n_qubits=2)
        gate = parse_gate_spec('h0', n_qubits=2)

        graph.apply_layer([gate])

        # Should add only one timestep
        assert graph.n_timesteps == 1

        # Final state should have superposition on qubit 0
        final_state = graph.get_final_state()
        assert '00' in final_state
        assert '10' in final_state

    def test_apply_layer_multiple_gates_sequential(self):
        """Test applying multiple gates in a single layer."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec

        graph = FeynmanGraph(n_qubits=4)
        gates = [
            parse_gate_spec('h0', n_qubits=4),
            parse_gate_spec('h1', n_qubits=4),
            parse_gate_spec('h2', n_qubits=4),
            parse_gate_spec('h3', n_qubits=4)
        ]

        graph.apply_layer(gates)

        # Should add only one timestep (not 4)
        assert graph.n_timesteps == 1

        # Final state should be superposition of all 16 states
        final_state = graph.get_final_state()
        assert len(final_state) == 16  # 2^4 states

        # All amplitudes should be 1/4 (1/2^4)
        expected_amp = 0.25
        for state, amp in final_state.items():
            assert abs(abs(amp) - expected_amp) < 1e-10

    def test_apply_layer_produces_cumulative_amplitudes(self):
        """Test that layer produces cumulative amplitudes in JSON output."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec

        graph = FeynmanGraph(n_qubits=2)
        gates = [
            parse_gate_spec('h0', n_qubits=2),
            parse_gate_spec('cnot0,1', n_qubits=2)
        ]

        graph.apply_layer(gates)

        json_data = graph.to_json()

        # Should have 2 columns (layer + final)
        assert len(json_data['cols']) == 2

        # First column should show transitions from |00⟩ to final states
        col0 = json_data['cols'][0]
        assert '00' in col0
        assert col0['00']['amp'] == 1  # Initial amplitude

        # After H0 and CNOT(0,1), should have |00⟩ and |11⟩
        next_states = col0['00']['next']
        assert '00' in next_states
        assert '11' in next_states

        # Cumulative amplitudes should be 1/√2
        s2_2 = 1 / np.sqrt(2)
        assert abs(next_states['00'] - s2_2) < 1e-10
        assert abs(next_states['11'] - s2_2) < 1e-10

    def test_multiple_layers_sequential(self):
        """Test applying multiple layers in sequence."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec

        graph = FeynmanGraph(n_qubits=2)

        # Layer 1: H on both qubits
        layer1 = [
            parse_gate_spec('h0', n_qubits=2),
            parse_gate_spec('h1', n_qubits=2)
        ]
        graph.apply_layer(layer1)

        # Layer 2: CNOT
        layer2 = [parse_gate_spec('cnot0,1', n_qubits=2)]
        graph.apply_layer(layer2)

        # Layer 3: Z on qubit 1
        layer3 = [parse_gate_spec('z1', n_qubits=2)]
        graph.apply_layer(layer3)

        # Should have 3 timesteps (one per layer)
        assert graph.n_timesteps == 3

        json_data = graph.to_json()
        # Should have 4 columns (3 layers + final)
        assert len(json_data['cols']) == 4

    def test_apply_layer_empty_list_raises_error(self):
        """Test that empty layer raises error."""
        from feynman_path.core.graph import FeynmanGraph

        graph = FeynmanGraph(n_qubits=2)

        with pytest.raises(ValueError, match="empty layer"):
            graph.apply_layer([])


class TestLayerGateEquivalence:
    """Test that layer-based output matches gate-by-gate output."""

    def test_layer_final_state_matches_gate_by_gate(self):
        """Test that final state is identical for layer vs gate-by-gate."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec

        # Gate-by-gate execution
        graph1 = FeynmanGraph(n_qubits=2)
        for gate_str in ['h0', 'h1', 'cnot0,1', 'z1']:
            gate = parse_gate_spec(gate_str, n_qubits=2)
            graph1.apply_gate(gate)

        # Layer execution (all gates in one layer)
        graph2 = FeynmanGraph(n_qubits=2)
        gates = [parse_gate_spec(g, n_qubits=2) for g in ['h0', 'h1', 'cnot0,1', 'z1']]
        graph2.apply_layer(gates)

        # Final states should be identical
        final1 = graph1.get_final_state()
        final2 = graph2.get_final_state()

        assert set(final1.keys()) == set(final2.keys())
        for state in final1:
            assert abs(final1[state] - final2[state]) < 1e-10

    def test_multi_layer_intermediate_states_match(self):
        """Test that intermediate states match at corresponding timesteps."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec

        # Gate-by-gate: h0 h1 | cnot0,1 | z1
        graph1 = FeynmanGraph(n_qubits=2)
        for gate_str in ['h0', 'h1']:
            graph1.apply_gate(parse_gate_spec(gate_str, n_qubits=2))
        state_after_layer1 = graph1.get_final_state().copy()

        graph1.apply_gate(parse_gate_spec('cnot0,1', n_qubits=2))
        state_after_layer2 = graph1.get_final_state().copy()

        graph1.apply_gate(parse_gate_spec('z1', n_qubits=2))
        state_after_layer3 = graph1.get_final_state().copy()

        # Layer mode
        graph2 = FeynmanGraph(n_qubits=2)
        graph2.apply_layer([parse_gate_spec(g, n_qubits=2) for g in ['h0', 'h1']])
        layer_state1 = graph2.get_final_state().copy()

        graph2.apply_layer([parse_gate_spec('cnot0,1', n_qubits=2)])
        layer_state2 = graph2.get_final_state().copy()

        graph2.apply_layer([parse_gate_spec('z1', n_qubits=2)])
        layer_state3 = graph2.get_final_state().copy()

        # Compare states at each corresponding timestep
        def states_equal(s1, s2):
            if set(s1.keys()) != set(s2.keys()):
                return False
            for state in s1:
                if abs(s1[state] - s2[state]) >= 1e-10:
                    return False
            return True

        assert states_equal(state_after_layer1, layer_state1)
        assert states_equal(state_after_layer2, layer_state2)
        assert states_equal(state_after_layer3, layer_state3)

    def test_bell_pair_layer_equivalence(self):
        """Test Bell pair creation: layer vs gate-by-gate."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec

        # Gate-by-gate
        graph1 = FeynmanGraph(n_qubits=2)
        graph1.apply_gate(parse_gate_spec('h0', n_qubits=2))
        graph1.apply_gate(parse_gate_spec('cnot0,1', n_qubits=2))

        # Layer mode
        graph2 = FeynmanGraph(n_qubits=2)
        gates = [parse_gate_spec(g, n_qubits=2) for g in ['h0', 'cnot0,1']]
        graph2.apply_layer(gates)

        # Final states should match (Bell pair: |00⟩ + |11⟩)
        final1 = graph1.get_final_state()
        final2 = graph2.get_final_state()

        assert set(final1.keys()) == set(final2.keys())
        assert set(final1.keys()) == {'00', '11'}

        for state in final1:
            assert abs(final1[state] - final2[state]) < 1e-10

    def test_complex_circuit_layer_equivalence(self):
        """Test the interference circuit from README."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec

        gates_str = ['h0', 'cnot0,1', 'z1', 'h0', 'h1', 'cnot1,0', 'h1']

        # Gate-by-gate
        graph1 = FeynmanGraph(n_qubits=2)
        for gate_str in gates_str:
            graph1.apply_gate(parse_gate_spec(gate_str, n_qubits=2))

        # Layer mode (all in one layer)
        graph2 = FeynmanGraph(n_qubits=2)
        gates = [parse_gate_spec(g, n_qubits=2) for g in gates_str]
        graph2.apply_layer(gates)

        # Final states should be identical
        final1 = graph1.get_final_state()
        final2 = graph2.get_final_state()

        assert set(final1.keys()) == set(final2.keys())
        for state in final1:
            assert abs(final1[state] - final2[state]) < 1e-10

    def test_parametrized_gates_in_layer(self):
        """Test that rotation gates work correctly in layers."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec

        # Gate-by-gate
        graph1 = FeynmanGraph(n_qubits=2)
        graph1.apply_gate(parse_gate_spec('rx0,pi/4', n_qubits=2))
        graph1.apply_gate(parse_gate_spec('ry1,pi/2', n_qubits=2))

        # Layer mode
        graph2 = FeynmanGraph(n_qubits=2)
        gates = [
            parse_gate_spec('rx0,pi/4', n_qubits=2),
            parse_gate_spec('ry1,pi/2', n_qubits=2)
        ]
        graph2.apply_layer(gates)

        # Final states should match
        final1 = graph1.get_final_state()
        final2 = graph2.get_final_state()

        assert set(final1.keys()) == set(final2.keys())
        for state in final1:
            assert abs(final1[state] - final2[state]) < 1e-10

    def test_multi_controlled_gates_in_layer(self):
        """Test that multi-controlled gates work in layers."""
        from feynman_path.core.graph import FeynmanGraph
        from feynman_path.core.gates import parse_gate_spec

        # Gate-by-gate
        graph1 = FeynmanGraph(n_qubits=3)
        graph1.apply_gate(parse_gate_spec('h0', n_qubits=3))
        graph1.apply_gate(parse_gate_spec('h1', n_qubits=3))
        graph1.apply_gate(parse_gate_spec('toffoli0,1,2', n_qubits=3))

        # Layer mode
        graph2 = FeynmanGraph(n_qubits=3)
        gates = [
            parse_gate_spec('h0', n_qubits=3),
            parse_gate_spec('h1', n_qubits=3),
            parse_gate_spec('toffoli0,1,2', n_qubits=3)
        ]
        graph2.apply_layer(gates)

        # Final states should match
        final1 = graph1.get_final_state()
        final2 = graph2.get_final_state()

        assert set(final1.keys()) == set(final2.keys())
        for state in final1:
            assert abs(final1[state] - final2[state]) < 1e-10
