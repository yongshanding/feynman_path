"""Gate definitions and operations for quantum circuits."""

from typing import Dict, Tuple, Union, Callable
import numpy as np
import numpy.typing as npt


# Standard gate matrices
GATE_MATRICES: Dict[str, npt.NDArray[np.complex128]] = {
    'H': np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2),
    'X': np.array([[0, 1], [1, 0]], dtype=np.complex128),
    'Z': np.array([[1, 0], [0, -1]], dtype=np.complex128),
    'S': np.array([[1, 0], [0, 1j]], dtype=np.complex128),
    'CNOT': np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ], dtype=np.complex128),
    'CONOT': np.array([  # 0-controlled NOT
        [0, 1, 0, 0],
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.complex128),
}


class Gate:
    """
    Quantum gate with matrix representation.

    Supports both standard gates and custom user-defined unitaries.
    """

    def __init__(
        self,
        name: str,
        matrix: npt.NDArray[np.complex128],
        qubits: Tuple[int, ...],
        n_qubits: int,
        validate: bool = False
    ):
        """
        Initialize quantum gate.

        Args:
            name: Gate name (e.g., 'H', 'CNOT', 'CustomGate')
            matrix: Unitary matrix representation
            qubits: Tuple of qubit indices this gate acts on
            n_qubits: Total number of qubits in the system
            validate: If True, check that matrix is unitary

        Raises:
            ValueError: If validation enabled and matrix is not unitary
        """
        self.name = name
        self.matrix = np.array(matrix, dtype=np.complex128)
        self.qubits = qubits
        self.n_qubits = n_qubits

        if validate:
            self._validate_unitary()

    def _validate_unitary(self) -> None:
        """
        Check that gate matrix is unitary: U† U = I.

        Raises:
            ValueError: If matrix is not unitary
        """
        product = self.matrix @ self.matrix.conj().T
        identity = np.eye(len(self.matrix))

        if not np.allclose(product, identity, atol=1e-10):
            raise ValueError(
                f"Gate matrix for '{self.name}' is not unitary. "
                f"U† U must equal identity."
            )

    def apply_to_state(self, state: Union[str, int]) -> Dict[str, complex]:
        """
        Apply gate to a basis state and return all transitions.

        This is the core of the Feynman path calculation - it computes
        which states this gate can transition to from the input state,
        and with what amplitude (edge weight in the Feynman graph).

        Args:
            state: Input state as bitstring ('00') or integer (0)

        Returns:
            Dictionary mapping output states to transition amplitudes:
            {'00': 0.707, '10': 0.707} means transitions to |00⟩ and |10⟩
        """
        # Convert state to integer if needed
        state_int = self._state_to_int(state)

        # Extract relevant qubits for this gate
        relevant_bits = self._extract_qubits(state_int, self.qubits)

        # Get other qubits (spectator qubits not affected by this gate)
        spectator_bits = self._get_spectator_bits(state_int, self.qubits)

        # Apply gate matrix to relevant qubits
        # Create input vector for these qubits
        dim = 2 ** len(self.qubits)
        input_vector = np.zeros(dim, dtype=np.complex128)
        input_vector[relevant_bits] = 1.0

        # Matrix multiplication
        output_vector = self.matrix @ input_vector

        # Build output states by combining spectators with gate outputs
        transitions = {}
        for output_idx, amplitude in enumerate(output_vector):
            if abs(amplitude) > 1e-10:  # Filter negligible amplitudes
                # Combine spectator bits with new output bits
                output_state_int = self._combine_bits(
                    spectator_bits, output_idx, self.qubits
                )
                output_state_str = self._int_to_bitstring(output_state_int)
                transitions[output_state_str] = float(amplitude) if amplitude.imag == 0 else complex(amplitude)

        return transitions

    def _state_to_int(self, state: Union[str, int]) -> int:
        """Convert bitstring to integer."""
        if isinstance(state, int):
            return state
        # Reverse for qubit ordering (LSB = qubit 0)
        return int(state[::-1], 2)

    def _int_to_bitstring(self, state_int: int) -> str:
        """Convert integer to bitstring."""
        bitstring = format(state_int, f'0{self.n_qubits}b')
        return bitstring[::-1]  # Reverse for qubit ordering

    def _extract_qubits(self, state_int: int, qubit_indices: Tuple[int, ...]) -> int:
        """
        Extract specific qubit values from state in matrix basis ordering.

        The matrix uses standard big-endian ordering where the first qubit
        in qubit_indices corresponds to the most significant bit.

        E.g., qubits=(0,1), state '10' (qubit0=1, qubit1=0):
        - Extract [qubit0, qubit1] = [1, 0]
        - Matrix basis: |qubit0, qubit1⟩ = |1,0⟩ = 0b10 = 2
        """
        result = 0
        n = len(qubit_indices)
        for i, qubit_idx in enumerate(qubit_indices):
            bit = (state_int >> qubit_idx) & 1
            # Place in big-endian position: first qubit → MSB
            result |= (bit << (n - 1 - i))
        return result

    def _get_spectator_bits(self, state_int: int, active_qubits: Tuple[int, ...]) -> Dict[int, int]:
        """Get bits of qubits not involved in this gate."""
        spectators = {}
        for qubit_idx in range(self.n_qubits):
            if qubit_idx not in active_qubits:
                bit = (state_int >> qubit_idx) & 1
                spectators[qubit_idx] = bit
        return spectators

    def _combine_bits(
        self,
        spectator_bits: Dict[int, int],
        gate_output: int,
        active_qubits: Tuple[int, ...]
    ) -> int:
        """Combine spectator bits with gate output bits to form full state.

        gate_output is in big-endian matrix basis ordering, need to convert back.
        """
        result = 0

        # Set spectator bits
        for qubit_idx, bit in spectator_bits.items():
            result |= (bit << qubit_idx)

        # Set active qubit bits from gate output (converting from big-endian)
        n = len(active_qubits)
        for i, qubit_idx in enumerate(active_qubits):
            # Extract bit from big-endian position
            bit = (gate_output >> (n - 1 - i)) & 1
            result |= (bit << qubit_idx)

        return result


# Gate factory functions for registry
def _make_single_qubit_gate(gate_name: str) -> Callable:
    """Factory for single-qubit gates."""
    def create(qubit: int, n_qubits: int) -> Gate:
        return Gate(
            name=gate_name.upper(),
            matrix=GATE_MATRICES[gate_name.upper()],
            qubits=(qubit,),
            n_qubits=n_qubits
        )
    return create


def _make_cnot_gate(control: int, target: int, n_qubits: int) -> Gate:
    """Create CNOT gate."""
    return Gate(
        name='CNOT',
        matrix=GATE_MATRICES['CNOT'],
        qubits=(control, target),
        n_qubits=n_qubits
    )


def _make_conot_gate(control: int, target: int, n_qubits: int) -> Gate:
    """Create CONOT (0-controlled NOT) gate."""
    return Gate(
        name='CONOT',
        matrix=GATE_MATRICES['CONOT'],
        qubits=(control, target),
        n_qubits=n_qubits
    )


# Gate registry for parsing gate specifications
GATE_REGISTRY: Dict[str, Callable] = {
    'h': _make_single_qubit_gate('H'),
    'x': _make_single_qubit_gate('X'),
    'z': _make_single_qubit_gate('Z'),
    's': _make_single_qubit_gate('S'),
    'cnot': _make_cnot_gate,
    'conot': _make_conot_gate,
}


def parse_gate_spec(gate_spec: str, n_qubits: int) -> Gate:
    """
    Parse gate specification string into Gate object.

    Args:
        gate_spec: Gate specification like 'h0', 'cnot0,1', 'z2'
        n_qubits: Total number of qubits in the circuit

    Returns:
        Gate object

    Raises:
        ValueError: If gate specification is invalid
        KeyError: If gate name is not recognized

    Examples:
        >>> parse_gate_spec('h0', n_qubits=2)  # Hadamard on qubit 0
        >>> parse_gate_spec('cnot0,1', n_qubits=2)  # CNOT control=0, target=1
        >>> parse_gate_spec('z1', n_qubits=3)  # Z gate on qubit 1
    """
    # Extract gate name and qubit indices
    # Format: gatename + qubits (e.g., 'h0', 'cnot0,1')
    name_part = ''
    for char in gate_spec:
        if char.isalpha():
            name_part += char
        else:
            break

    gate_name = name_part.lower()
    qubit_part = gate_spec[len(name_part):]

    if gate_name not in GATE_REGISTRY:
        raise KeyError(
            f"Unknown gate '{gate_name}'. "
            f"Available gates: {list(GATE_REGISTRY.keys())}"
        )

    # Parse qubit indices
    if ',' in qubit_part:
        # Two-qubit gate
        qubit_strs = qubit_part.split(',')
        qubits = [int(q) for q in qubit_strs]

        # Validate qubit indices
        for q in qubits:
            if q < 0 or q >= n_qubits:
                raise ValueError(
                    f"Qubit index {q} out of range [0, {n_qubits-1}]"
                )

        gate = GATE_REGISTRY[gate_name](*qubits, n_qubits=n_qubits)
    else:
        # Single-qubit gate
        qubit = int(qubit_part)

        if qubit < 0 or qubit >= n_qubits:
            raise ValueError(
                f"Qubit index {qubit} out of range [0, {n_qubits-1}]"
            )

        gate = GATE_REGISTRY[gate_name](qubit=qubit, n_qubits=n_qubits)

    return gate
