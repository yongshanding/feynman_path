"""Gate definitions and operations for quantum circuits."""

from typing import Dict, Tuple, Union, Callable, Optional
import numpy as np
import numpy.typing as npt
import re


# Standard gate matrices
GATE_MATRICES: Dict[str, npt.NDArray[np.complex128]] = {
    'H': np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2),
    'X': np.array([[0, 1], [1, 0]], dtype=np.complex128),
    'Y': np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
    'Z': np.array([[1, 0], [0, -1]], dtype=np.complex128),
    'S': np.array([[1, 0], [0, 1j]], dtype=np.complex128),
    'Sdag': np.array([[1, 0], [0, -1j]], dtype=np.complex128),
    'T': np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128),
    'Tdag': np.array([[1, 0], [0, np.exp(-1j * np.pi / 4)]], dtype=np.complex128),
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
    'Toffoli': np.array([  # Controlled-Controlled-NOT (CCX, CCNOT)
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1],  # |110⟩ → |111⟩
        [0, 0, 0, 0, 0, 0, 1, 0],  # |111⟩ → |110⟩
    ], dtype=np.complex128),
}


# Rotation gate matrix generation functions
def generate_rx_matrix(theta: float) -> npt.NDArray[np.complex128]:
    """
    Generate Rx(θ) rotation matrix.

    Rx(θ) = cos(θ/2)*I - i*sin(θ/2)*X

    Args:
        theta: Rotation angle in radians

    Returns:
        2x2 rotation matrix
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([
        [c, -1j*s],
        [-1j*s, c]
    ], dtype=np.complex128)


def generate_ry_matrix(theta: float) -> npt.NDArray[np.complex128]:
    """
    Generate Ry(θ) rotation matrix.

    Ry(θ) = cos(θ/2)*I - i*sin(θ/2)*Y

    Args:
        theta: Rotation angle in radians

    Returns:
        2x2 rotation matrix
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([
        [c, -s],
        [s, c]
    ], dtype=np.complex128)


def generate_rz_matrix(theta: float) -> npt.NDArray[np.complex128]:
    """
    Generate Rz(θ) rotation matrix.

    Rz(θ) = cos(θ/2)*I - i*sin(θ/2)*Z

    Args:
        theta: Rotation angle in radians

    Returns:
        2x2 rotation matrix
    """
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([
        [c - 1j*s, 0],
        [0, c + 1j*s]
    ], dtype=np.complex128)


def parse_angle(angle_str: str) -> float:
    """
    Parse angle string to float value.

    Supports:
    - Numeric values: '1.5708', '0.785'
    - 'pi': 3.14159...
    - Expressions: 'pi/2', '2*pi/3', 'pi/4'

    Args:
        angle_str: String representing the angle

    Returns:
        Angle value in radians
    """
    # Replace 'pi' with np.pi value for evaluation
    angle_str = angle_str.strip()

    # Simple numeric value
    try:
        return float(angle_str)
    except ValueError:
        pass

    # Expression containing 'pi'
    if 'pi' in angle_str:
        # Replace 'pi' with the numeric value
        expr = angle_str.replace('pi', str(np.pi))
        try:
            return float(eval(expr, {"__builtins__": {}}, {}))
        except:
            raise ValueError(f"Could not parse angle expression: '{angle_str}'")

    raise ValueError(f"Invalid angle format: '{angle_str}'")


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
        validate: bool = False,
        angle: Optional[float] = None
    ):
        """
        Initialize quantum gate.

        Args:
            name: Gate name (e.g., 'H', 'CNOT', 'CustomGate', 'Rx')
            matrix: Unitary matrix representation
            qubits: Tuple of qubit indices this gate acts on
            n_qubits: Total number of qubits in the system
            validate: If True, check that matrix is unitary
            angle: Rotation angle in radians (for parametrized gates like Rx, Ry, Rz)

        Raises:
            ValueError: If validation enabled and matrix is not unitary
        """
        self.name = name
        self.matrix = np.array(matrix, dtype=np.complex128)
        self.qubits = qubits
        self.n_qubits = n_qubits
        self.angle = angle  # Store angle for parametrized gates

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


class MultiControlledGate(Gate):
    """
    Multi-controlled gate using sparse representation.

    Instead of generating exponentially large matrices for k controls,
    this class uses conditional logic to check if all control qubits are 1,
    and only then applies the operation to the target qubit.
    """

    def __init__(
        self,
        name: str,
        control_qubits: Tuple[int, ...],
        target_qubit: int,
        n_qubits: int
    ):
        """
        Initialize multi-controlled gate.

        Args:
            name: Gate name (e.g., 'Toffoli', 'M3CNOT')
            control_qubits: Tuple of control qubit indices
            target_qubit: Target qubit index
            n_qubits: Total number of qubits in the system
        """
        # Create a dummy 2x2 identity matrix (not used for computation)
        dummy_matrix = np.eye(2, dtype=np.complex128)

        # Combine control and target qubits
        all_qubits = control_qubits + (target_qubit,)

        # Initialize parent with dummy matrix
        super().__init__(
            name=name,
            matrix=dummy_matrix,
            qubits=all_qubits,
            n_qubits=n_qubits,
            validate=False
        )

        self.control_qubits = control_qubits
        self.target_qubit = target_qubit

    def apply_to_state(self, state: Union[str, int]) -> Dict[str, complex]:
        """
        Apply multi-controlled gate using sparse representation.

        Checks if all control qubits are 1. If so, flips target qubit.
        Otherwise, returns the input state unchanged.

        Args:
            state: Input state as bitstring ('00') or integer (0)

        Returns:
            Dictionary mapping output states to transition amplitudes
        """
        # Convert state to integer
        state_int = self._state_to_int(state)

        # Check if all control qubits are 1
        all_controls_one = True
        for control_qubit in self.control_qubits:
            bit = (state_int >> control_qubit) & 1
            if bit != 1:
                all_controls_one = False
                break

        if all_controls_one:
            # Flip target qubit
            output_state_int = state_int ^ (1 << self.target_qubit)
            output_state_str = self._int_to_bitstring(output_state_int)
            return {output_state_str: 1.0}
        else:
            # Return unchanged state
            input_state_str = self._int_to_bitstring(state_int)
            return {input_state_str: 1.0}


# Gate factory functions for registry
def _make_single_qubit_gate(gate_name: str) -> Callable:
    """Factory for single-qubit gates."""
    def create(qubit: int, n_qubits: int) -> Gate:
        return Gate(
            name=gate_name,  # Use provided name as-is (already in correct case)
            matrix=GATE_MATRICES[gate_name],
            qubits=(qubit,),
            n_qubits=n_qubits
        )
    return create


def _make_cnot_gate(control: int, target: int, n_qubits: int) -> Gate:
    """Create CNOT gate."""
    if control == target:
        raise ValueError(
            f"CNOT gate cannot have same control and target qubit. "
            f"Got: control={control}, target={target}"
        )
    return Gate(
        name='CNOT',
        matrix=GATE_MATRICES['CNOT'],
        qubits=(control, target),
        n_qubits=n_qubits
    )


def _make_conot_gate(control: int, target: int, n_qubits: int) -> Gate:
    """Create CONOT (0-controlled NOT) gate."""
    if control == target:
        raise ValueError(
            f"CONOT gate cannot have same control and target qubit. "
            f"Got: control={control}, target={target}"
        )
    return Gate(
        name='CONOT',
        matrix=GATE_MATRICES['CONOT'],
        qubits=(control, target),
        n_qubits=n_qubits
    )


def _make_toffoli_gate(control0: int, control1: int, target: int, n_qubits: int) -> MultiControlledGate:
    """Create Toffoli (CCX/CCNOT) gate using sparse representation."""
    # Check for repeated qubits
    all_qubits = [control0, control1, target]
    if len(all_qubits) != len(set(all_qubits)):
        raise ValueError(
            f"Toffoli gate cannot have repeated qubits. "
            f"Got: control0={control0}, control1={control1}, target={target}"
        )

    return MultiControlledGate(
        name='Toffoli',
        control_qubits=(control0, control1),
        target_qubit=target,
        n_qubits=n_qubits
    )


def _make_mcnot_gate(k: int, *qubits, n_qubits: int) -> MultiControlledGate:
    """
    Create multi-controlled NOT gate with k controls using sparse representation.

    Args:
        k: Number of control qubits (1 <= k <= 15)
        *qubits: k control qubits followed by 1 target qubit
        n_qubits: Total number of qubits in the system

    Returns:
        MultiControlledGate instance

    Raises:
        ValueError: If k is out of bounds, wrong number of qubits provided, or repeated qubits
    """
    # Validate k
    if k < 1 or k > 15:
        raise ValueError(
            f"Number of controls k={k} is out of bounds. "
            f"Must be between 1 and 15 inclusive."
        )

    # Validate number of qubits provided
    if len(qubits) != k + 1:
        raise ValueError(
            f"m{k}cnot requires {k} control qubits + 1 target qubit "
            f"({k+1} total), but got {len(qubits)} qubits"
        )

    # Check for repeated qubits
    if len(qubits) != len(set(qubits)):
        raise ValueError(
            f"m{k}cnot gate cannot have repeated qubits. "
            f"Got qubits: {qubits}"
        )

    control_qubits = qubits[:k]
    target_qubit = qubits[k]

    return MultiControlledGate(
        name=f'M{k}CNOT',
        control_qubits=control_qubits,
        target_qubit=target_qubit,
        n_qubits=n_qubits
    )


# Gate registry for parsing gate specifications
GATE_REGISTRY: Dict[str, Callable] = {
    'h': _make_single_qubit_gate('H'),
    'x': _make_single_qubit_gate('X'),
    'y': _make_single_qubit_gate('Y'),
    'z': _make_single_qubit_gate('Z'),
    's': _make_single_qubit_gate('S'),
    'sdag': _make_single_qubit_gate('Sdag'),
    't': _make_single_qubit_gate('T'),
    'tdag': _make_single_qubit_gate('Tdag'),
    'cnot': _make_cnot_gate,
    'conot': _make_conot_gate,
    'toffoli': _make_toffoli_gate,
}


def parse_gate_spec(gate_spec: str, n_qubits: int) -> Gate:
    """
    Parse gate specification string into Gate object.

    Args:
        gate_spec: Gate specification like 'h0', 'cnot0,1', 'rx0,pi/4', 'toffoli0,1,2', 'm3cnot0,1,2,3'
        n_qubits: Total number of qubits in the circuit

    Returns:
        Gate object

    Raises:
        ValueError: If gate specification is invalid
        KeyError: If gate name is not recognized

    Examples:
        >>> parse_gate_spec('h0', n_qubits=2)  # Hadamard on qubit 0
        >>> parse_gate_spec('cnot0,1', n_qubits=2)  # CNOT control=0, target=1
        >>> parse_gate_spec('rx0,pi/4', n_qubits=2)  # Rx rotation on qubit 0
        >>> parse_gate_spec('toffoli0,1,2', n_qubits=3)  # Toffoli gate
        >>> parse_gate_spec('m3cnot0,1,2,3', n_qubits=4)  # 3-controlled NOT
    """
    # Check for m[k]cnot pattern first (e.g., 'm3cnot0,1,2,3')
    mcnot_match = re.match(r'^m(\d+)cnot(.+)$', gate_spec.lower())
    if mcnot_match:
        k = int(mcnot_match.group(1))
        params_part = mcnot_match.group(2)

        # Parse qubit indices
        qubit_strs = params_part.split(',')
        qubits = [int(q) for q in qubit_strs]

        # Validate qubit indices
        for q in qubits:
            if q < 0 or q >= n_qubits:
                raise ValueError(
                    f"Qubit index {q} out of range [0, {n_qubits-1}]"
                )

        # Create m[k]cnot gate
        return _make_mcnot_gate(k, *qubits, n_qubits=n_qubits)

    # Extract gate name for standard gates
    name_part = ''
    for char in gate_spec:
        if char.isalpha():
            name_part += char
        else:
            break

    gate_name = name_part.lower()
    params_part = gate_spec[len(name_part):]

    # Check if it's a parametrized rotation gate
    if gate_name in ['rx', 'ry', 'rz']:
        # Parse: qubit,angle (e.g., '0,pi/4')
        params = params_part.split(',', 1)  # Split into at most 2 parts
        if len(params) != 2:
            raise ValueError(
                f"Rotation gate '{gate_name}' requires format: {gate_name}qubit,angle"
            )

        qubit = int(params[0])
        angle = parse_angle(params[1])

        # Validate qubit index
        if qubit < 0 or qubit >= n_qubits:
            raise ValueError(
                f"Qubit index {qubit} out of range [0, {n_qubits-1}]"
            )

        # Generate the appropriate rotation matrix
        if gate_name == 'rx':
            matrix = generate_rx_matrix(angle)
            name = 'Rx'
        elif gate_name == 'ry':
            matrix = generate_ry_matrix(angle)
            name = 'Ry'
        else:  # rz
            matrix = generate_rz_matrix(angle)
            name = 'Rz'

        return Gate(
            name=name,
            matrix=matrix,
            qubits=(qubit,),
            n_qubits=n_qubits,
            angle=angle
        )

    # Standard (non-parametrized) gates
    if gate_name not in GATE_REGISTRY:
        raise KeyError(
            f"Unknown gate '{gate_name}'. "
            f"Available gates: {list(GATE_REGISTRY.keys())} and rx, ry, rz"
        )

    # Parse qubit indices
    if ',' in params_part:
        # Two-qubit gate
        qubit_strs = params_part.split(',')
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
        qubit = int(params_part)

        if qubit < 0 or qubit >= n_qubits:
            raise ValueError(
                f"Qubit index {qubit} out of range [0, {n_qubits-1}]"
            )

        gate = GATE_REGISTRY[gate_name](qubit=qubit, n_qubits=n_qubits)

    return gate
