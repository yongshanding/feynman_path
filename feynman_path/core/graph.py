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

            # Compute cumulative amplitudes for next states
            cumulative_next = {}
            for next_state_str, transition_amp in transitions.items():
                # Multiply current amplitude by transition amplitude
                cumulative_amp = amplitude * transition_amp
                cumulative_next[next_state_str] = cumulative_amp
                # Update next state amplitudes
                next_state.add(next_state_str, cumulative_amp)

            # Store cumulative amplitudes for JSON
            json_col[state_str] = {
                'amp': amplitude,
                'next': cumulative_next
            }

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
        # Build columns list starting with existing gate columns
        cols = list(self._timestep_data)

        # Add final layer with current state amplitudes and empty next
        final_col = {}
        for state_str, amplitude in self.current_state.items():
            final_col[state_str] = {
                'amp': amplitude,
                'next': {}
            }
        cols.append(final_col)

        return {
            'type': 'feynmanpath',
            'cols': cols
        }

    def apply_layer(self, gates: List[Gate]) -> None:
        """
        Apply multiple gates as a single layer.

        All gates in the layer are applied sequentially to the quantum state,
        but only one timestep column is recorded in the JSON output. This
        allows for layer-based visualization where each column represents
        a group of gates instead of individual gates.

        Args:
            gates: List of Gate objects to apply as a layer

        Raises:
            ValueError: If gates list is empty
        """
        if not gates:
            raise ValueError(
                "Cannot apply empty layer. "
                "Each layer must contain at least one gate."
            )

        # Store initial state for JSON column
        initial_states = {}
        for state_str, amplitude in self.current_state.items():
            initial_states[state_str] = amplitude

        # Apply all gates sequentially
        for gate in gates:
            if gate.n_qubits != self.n_qubits:
                raise ValueError(
                    f"Gate is for {gate.n_qubits} qubits but graph has {self.n_qubits}"
                )

            next_state = SparseStateVector(self.n_qubits)

            # Apply gate to current state
            for state_str, amplitude in self.current_state.items():
                transitions = gate.apply_to_state(state_str)
                for next_state_str, transition_amp in transitions.items():
                    cumulative_amp = amplitude * transition_amp
                    next_state.add(next_state_str, cumulative_amp)

            # Update current state for next gate in layer
            self.current_state = next_state

        # Now record the cumulative transition for JSON output
        json_col = {}
        for initial_state_str, initial_amplitude in initial_states.items():
            # Compute cumulative transitions from initial state to final states
            cumulative_next = {}

            # We need to track which final states came from this initial state
            # Apply all gates to this specific initial state
            temp_state = SparseStateVector(self.n_qubits)
            temp_state.set(initial_state_str, amplitude=1)

            for gate in gates:
                next_temp_state = SparseStateVector(self.n_qubits)
                for state_str, amplitude in temp_state.items():
                    transitions = gate.apply_to_state(state_str)
                    for next_state_str, transition_amp in transitions.items():
                        cumulative_amp = amplitude * transition_amp
                        next_temp_state.add(next_state_str, cumulative_amp)
                temp_state = next_temp_state

            # Now multiply by initial amplitude
            for final_state_str, transition_amp in temp_state.items():
                cumulative_next[final_state_str] = initial_amplitude * transition_amp

            json_col[initial_state_str] = {
                'amp': initial_amplitude,
                'next': cumulative_next
            }

        # Store this timestep's data
        self._timestep_data.append(json_col)
        self.n_timesteps += 1

    def get_final_state(self) -> Dict[str, Union[complex, float, sympy.Basic]]:
        """
        Get the final quantum state after all gates.

        Returns:
            Dictionary mapping state bitstrings to amplitudes
        """
        return self.current_state.to_dict()
