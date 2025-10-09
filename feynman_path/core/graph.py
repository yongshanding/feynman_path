"""Feynman path graph computation for quantum circuits."""

from typing import Dict, List, Union, Optional
import sympy
from .state import SparseStateVector
from .gates import Gate


class FeynmanGraph:
    """
    Computes Feynman path representation of a quantum circuit.

    The graph tracks all possible paths through the quantum computation,
    storing amplitudes and transitions at each timestep.
    """

    def __init__(
        self,
        n_qubits: int,
        initial_state: Optional[str] = None
    ):
        """
        Initialize Feynman path graph.

        Args:
            n_qubits: Number of qubits in the circuit
            initial_state: Initial state as bitstring (default: '00...0')
        """
        self.n_qubits = n_qubits

        # Set initial state (default to |00...0⟩)
        if initial_state is None:
            initial_state = '0' * n_qubits

        # Validate initial state
        if len(initial_state) != n_qubits:
            raise ValueError(
                f"Initial state length {len(initial_state)} doesn't match "
                f"n_qubits={n_qubits}"
            )

        # Initialize with single state at t=0
        self.current_state = SparseStateVector(n_qubits)
        self.current_state.set(initial_state, amplitude=1)

        # Store the history of states at each timestep for JSON output
        self._timestep_data: List[Dict] = []

        self.n_timesteps = 0

    def apply_gate(self, gate: Gate) -> None:
        """
        Apply a quantum gate to the graph.

        This computes all transitions from current states to next states
        and stores them for the Feynman path representation.

        Args:
            gate: Gate to apply
        """
        if gate.n_qubits != self.n_qubits:
            raise ValueError(
                f"Gate is for {gate.n_qubits} qubits but graph has {self.n_qubits}"
            )

        # Build the JSON column for this timestep
        json_col = {}
        next_state = SparseStateVector(self.n_qubits)

        # For each current state with non-zero amplitude
        for state_str, amplitude in self.current_state.items():
            # Get all possible transitions from this state
            transitions = gate.apply_to_state(state_str)

            # Store transitions for JSON
            json_col[state_str] = {
                'amp': amplitude,
                'next': transitions
            }

            # Update next state amplitudes
            for next_state_str, transition_amp in transitions.items():
                # Multiply current amplitude by transition amplitude
                new_amp = amplitude * transition_amp
                next_state.add(next_state_str, new_amp)

        # Store this timestep's data
        self._timestep_data.append(json_col)

        # Move to next timestep
        self.current_state = next_state
        self.n_timesteps += 1

    def to_json(self) -> Dict:
        """
        Export Feynman path graph to JSON format.

        Returns:
            Dictionary in the format:
            {
                'type': 'feynmanpath',
                'cols': [
                    {
                        'state': {'amp': amplitude, 'next': {next_states...}},
                        ...
                    },
                    ...
                ]
            }
        """
        return {
            'type': 'feynmanpath',
            'cols': self._timestep_data
        }

    def get_final_state(self) -> Dict[str, Union[complex, float, sympy.Basic]]:
        """
        Get the final quantum state after all gates.

        Returns:
            Dictionary mapping state bitstrings to amplitudes
        """
        return self.current_state.to_dict()
