"""Efficient sparse state vector representation for quantum states."""

from typing import Dict, Union, Iterator, Tuple
import sympy


class SparseStateVector:
    """
    Sparse representation of quantum state vectors.

    Only stores states with non-zero amplitudes to enable efficient
    computation for large qubit systems.

    Attributes:
        n_qubits: Number of qubits in the system
    """

    def __init__(self, n_qubits: int):
        """
        Initialize sparse state vector.

        Args:
            n_qubits: Number of qubits in the system
        """
        self.n_qubits = n_qubits
        self._states: Dict[int, Union[complex, float, sympy.Basic]] = {}
        self._amplitude_threshold = 1e-10  # Filter out tiny amplitudes

    def set(self, state: Union[str, int], amplitude: Union[complex, float, sympy.Basic]) -> None:
        """
        Set amplitude for a quantum state.

        Args:
            state: State as bitstring ('00', '11') or integer (0, 3)
            amplitude: Complex, float, or sympy amplitude value
        """
        state_int = self._to_int(state)

        # Filter out negligible amplitudes for numeric types
        if isinstance(amplitude, (int, float, complex)):
            if abs(amplitude) < self._amplitude_threshold:
                # Remove from storage if exists
                self._states.pop(state_int, None)
                return

        self._states[state_int] = amplitude

    def get(self, state: Union[str, int]) -> Union[complex, float, sympy.Basic]:
        """
        Get amplitude for a quantum state.

        Args:
            state: State as bitstring ('00') or integer (0)

        Returns:
            Amplitude value, or 0 if state not present
        """
        state_int = self._to_int(state)
        return self._states.get(state_int, 0)

    def add(self, state: Union[str, int], amplitude: Union[complex, float, sympy.Basic]) -> None:
        """
        Add amplitude to existing state (for superposition).

        Args:
            state: State as bitstring or integer
            amplitude: Amplitude to add to current value
        """
        state_int = self._to_int(state)
        current = self._states.get(state_int, 0)
        new_amp = current + amplitude
        self.set(state_int, new_amp)

    def clear(self) -> None:
        """Clear all state amplitudes."""
        self._states.clear()

    def to_dict(self) -> Dict[str, Union[complex, float, sympy.Basic]]:
        """
        Convert to dictionary with bitstring keys for JSON output.

        Returns:
            Dictionary mapping bitstrings to amplitudes: {'00': 1, '11': 0.5}
        """
        return {
            self._to_bitstring(state_int): amp
            for state_int, amp in self._states.items()
        }

    def items(self) -> Iterator[Tuple[str, Union[complex, float, sympy.Basic]]]:
        """
        Iterate over (bitstring, amplitude) pairs.

        Yields:
            Tuples of (bitstring, amplitude)
        """
        for state_int, amp in self._states.items():
            yield (self._to_bitstring(state_int), amp)

    def __len__(self) -> int:
        """Return number of states with non-zero amplitude."""
        return len(self._states)

    def __contains__(self, state: Union[str, int]) -> bool:
        """Check if state has non-zero amplitude."""
        state_int = self._to_int(state)
        return state_int in self._states

    def _to_int(self, state: Union[str, int]) -> int:
        """
        Convert state to integer representation.

        Args:
            state: Bitstring like '00' or integer like 0

        Returns:
            Integer representation

        Raises:
            ValueError: If bitstring has invalid format
        """
        if isinstance(state, int):
            return state

        # Validate bitstring
        if not isinstance(state, str):
            raise ValueError(f"State must be string or int, got {type(state)}")

        if len(state) != self.n_qubits:
            raise ValueError(
                f"Bitstring length {len(state)} doesn't match "
                f"n_qubits={self.n_qubits}"
            )

        if not all(c in '01' for c in state):
            raise ValueError(f"Bitstring must contain only '0' and '1', got '{state}'")

        # Convert binary string to integer (reversed for qubit ordering)
        return int(state[::-1], 2)

    def _to_bitstring(self, state_int: int) -> str:
        """
        Convert integer state to bitstring.

        Args:
            state_int: Integer representation of state

        Returns:
            Bitstring like '00', '11'
        """
        # Convert to binary, pad with zeros, reverse for qubit ordering
        bitstring = format(state_int, f'0{self.n_qubits}b')
        return bitstring[::-1]
