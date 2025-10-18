"""Tests for layer parsing functionality."""

import pytest


class TestLayerParsing:
    """Test parsing of gate lists into layers separated by '||'."""

    def test_parse_gates_no_separator_returns_individual_layers(self):
        """Test that gates without separator are each in their own layer (backward compatible)."""
        from feynman_path.core.layer_parser import parse_gate_layers

        gates = ['h0', 'cnot0,1', 'z1']
        result = parse_gate_layers(gates)

        # Each gate should be its own layer for backward compatibility
        assert result == [['h0'], ['cnot0,1'], ['z1']]

    def test_parse_gates_with_single_separator(self):
        """Test parsing with a single layer separator."""
        from feynman_path.core.layer_parser import parse_gate_layers

        gates = ['h0', 'h1', '-', 'cnot0,1']
        result = parse_gate_layers(gates)

        assert result == [['h0', 'h1'], ['cnot0,1']]

    def test_parse_gates_with_multiple_separators(self):
        """Test parsing with multiple layer separators."""
        from feynman_path.core.layer_parser import parse_gate_layers

        gates = ['h0', 'h1', '-', 'cnot0,1', 'cnot2,3', '-', 'cnot1,2', '-', 'h0']
        result = parse_gate_layers(gates)

        expected = [
            ['h0', 'h1'],
            ['cnot0,1', 'cnot2,3'],
            ['cnot1,2'],
            ['h0']
        ]
        assert result == expected

    def test_parse_gates_empty_layer_raises_error(self):
        """Test that consecutive separators (empty layer) raises error."""
        from feynman_path.core.layer_parser import parse_gate_layers

        gates = ['h0', '-', '-', 'h1']

        with pytest.raises(ValueError, match="empty layer"):
            parse_gate_layers(gates)

    def test_parse_gates_starts_with_separator_raises_error(self):
        """Test that starting with separator raises error."""
        from feynman_path.core.layer_parser import parse_gate_layers

        gates = ['-', 'h0', 'h1']

        with pytest.raises(ValueError, match="cannot start with"):
            parse_gate_layers(gates)

    def test_parse_gates_ends_with_separator_raises_error(self):
        """Test that ending with separator raises error."""
        from feynman_path.core.layer_parser import parse_gate_layers

        gates = ['h0', 'h1', '-']

        with pytest.raises(ValueError, match="cannot end with"):
            parse_gate_layers(gates)

    def test_parse_gates_empty_input_returns_empty_list(self):
        """Test that empty input returns empty list."""
        from feynman_path.core.layer_parser import parse_gate_layers

        gates = []
        result = parse_gate_layers(gates)

        assert result == []

    def test_parse_gates_single_gate(self):
        """Test parsing a single gate."""
        from feynman_path.core.layer_parser import parse_gate_layers

        gates = ['h0']
        result = parse_gate_layers(gates)

        assert result == [['h0']]

    def test_parse_gates_preserves_gate_strings(self):
        """Test that gate strings are preserved exactly."""
        from feynman_path.core.layer_parser import parse_gate_layers

        # Complex gate strings should be preserved
        gates = ['rx0,pi/4', 'ry1,1.5708', '-', 'm3cnot0,1,2,3', 'toffoli0,1,2']
        result = parse_gate_layers(gates)

        expected = [
            ['rx0,pi/4', 'ry1,1.5708'],
            ['m3cnot0,1,2,3', 'toffoli0,1,2']
        ]
        assert result == expected

    def test_parse_gates_whitespace_in_separator(self):
        """Test that only exact '-' is treated as separator."""
        from feynman_path.core.layer_parser import parse_gate_layers

        gates = ['h0', '-', 'h1']
        result = parse_gate_layers(gates)

        assert result == [['h0'], ['h1']]
