"""Tests for SparseStateVector class."""

import pytest
import sympy


class TestSparseStateVector:
    """Test sparse state vector representation."""

    def test_init_empty_state(self):
        """Test initialization of empty state vector."""
        from feynman_path.core.state import SparseStateVector

        state = SparseStateVector(n_qubits=2)
        assert state.n_qubits == 2
        assert len(state) == 0

    def test_set_state_with_bitstring(self):
        """Test setting state amplitude using bitstring."""
        from feynman_path.core.state import SparseStateVector

        state = SparseStateVector(n_qubits=2)
        state.set('00', amplitude=1)
        state.set('11', amplitude=0.5)

        assert len(state) == 2
        assert state.get('00') == 1
        assert state.get('11') == 0.5

    def test_set_state_with_integer(self):
        """Test setting state amplitude using integer."""
        from feynman_path.core.state import SparseStateVector

        state = SparseStateVector(n_qubits=2)
        state.set(0, amplitude=1)  # 0 = '00'
        state.set(3, amplitude=0.5)  # 3 = '11'

        assert state.get(0) == 1
        assert state.get(3) == 0.5
        assert state.get('00') == 1
        assert state.get('11') == 0.5

    def test_sparse_storage(self):
        """Test that only non-zero amplitudes are stored."""
        from feynman_path.core.state import SparseStateVector

        state = SparseStateVector(n_qubits=3)
        state.set('000', amplitude=1.0)
        state.set('001', amplitude=1e-12)  # Should be filtered out
        state.set('010', amplitude=0.5)

        # Only 2 states should be stored (filtering out tiny amplitude)
        assert len(state) == 2
        assert '000' in state
        assert '010' in state
        assert '001' not in state

    def test_to_dict_output_format(self):
        """Test conversion to dictionary with bitstring keys."""
        from feynman_path.core.state import SparseStateVector

        state = SparseStateVector(n_qubits=2)
        state.set(0, amplitude=1)
        state.set(3, amplitude=sympy.sqrt(2)/2)

        output = state.to_dict()

        assert isinstance(output, dict)
        assert '00' in output
        assert '11' in output
        assert output['00'] == 1
        assert output['11'] == sympy.sqrt(2)/2

    def test_iteration(self):
        """Test iterating over states."""
        from feynman_path.core.state import SparseStateVector

        state = SparseStateVector(n_qubits=2)
        state.set('00', amplitude=1)
        state.set('10', amplitude=0.5)

        states = list(state.items())
        assert len(states) == 2
        # Should return (bitstring, amplitude) pairs
        assert ('00', 1) in states or ('00', 1.0) in states
        assert ('10', 0.5) in states

    def test_bitstring_conversion(self):
        """Test bidirectional conversion between int and bitstring."""
        from feynman_path.core.state import SparseStateVector

        state = SparseStateVector(n_qubits=3)

        # Test int to bitstring
        assert state._to_bitstring(0) == '000'
        assert state._to_bitstring(7) == '111'
        assert state._to_bitstring(5) == '101'

        # Test bitstring to int
        assert state._to_int('000') == 0
        assert state._to_int('111') == 7
        assert state._to_int('101') == 5

    def test_add_to_existing_amplitude(self):
        """Test adding amplitude to existing state."""
        from feynman_path.core.state import SparseStateVector

        state = SparseStateVector(n_qubits=2)
        state.set('00', amplitude=0.5)
        state.add('00', amplitude=0.3)

        assert state.get('00') == 0.8

    def test_complex_amplitudes(self):
        """Test complex amplitude support."""
        from feynman_path.core.state import SparseStateVector

        state = SparseStateVector(n_qubits=2)
        state.set('00', amplitude=1j)
        state.set('11', amplitude=0.5 + 0.5j)

        assert state.get('00') == 1j
        assert state.get('11') == 0.5 + 0.5j

    def test_sympy_amplitudes(self):
        """Test sympy symbolic amplitude support."""
        from feynman_path.core.state import SparseStateVector

        state = SparseStateVector(n_qubits=2)
        amp = sympy.sqrt(2) / 2
        state.set('00', amplitude=amp)

        assert state.get('00') == amp
        assert isinstance(state.get('00'), sympy.Basic)

    def test_clear_state(self):
        """Test clearing all amplitudes."""
        from feynman_path.core.state import SparseStateVector

        state = SparseStateVector(n_qubits=2)
        state.set('00', amplitude=1)
        state.set('11', amplitude=0.5)

        state.clear()
        assert len(state) == 0

    def test_invalid_bitstring_length(self):
        """Test that invalid bitstring length raises error."""
        from feynman_path.core.state import SparseStateVector

        state = SparseStateVector(n_qubits=2)

        with pytest.raises((ValueError, AssertionError)):
            state.set('0', amplitude=1)  # Too short

        with pytest.raises((ValueError, AssertionError)):
            state.set('000', amplitude=1)  # Too long

    def test_invalid_bitstring_characters(self):
        """Test that non-binary characters raise error."""
        from feynman_path.core.state import SparseStateVector

        state = SparseStateVector(n_qubits=2)

        with pytest.raises((ValueError, AssertionError)):
            state.set('02', amplitude=1)  # Invalid character '2'
