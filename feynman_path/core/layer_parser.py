"""Parse gate specifications into layers separated by '-'."""

from typing import List

LAYER_SEPARATOR = '-'


def parse_gate_layers(gates: List[str]) -> List[List[str]]:
    """
    Parse a list of gate specifications into layers.

    Gates are grouped into layers separated by the '-' separator.
    Each layer will be applied as a group, with output showing one column
    per layer instead of one column per gate.

    WITHOUT '-': Each gate is its own layer (backward compatible, gate-by-gate mode)
    WITH '-': Gates between separators form layers

    Args:
        gates: List of gate specification strings, possibly containing '-' separators.
               Example: ['h0', 'h1', '-', 'cnot0,1', 'cnot2,3', '-', 'h0']

    Returns:
        List of layers, where each layer is a list of gate strings.
        Without separators: [['h0'], ['h1'], ['cnot0,1']] (gate-by-gate)
        With separators: [['h0', 'h1'], ['cnot0,1', 'cnot2,3'], ['h0']]

    Raises:
        ValueError: If the gate list has invalid structure:
            - Starts with separator
            - Ends with separator
            - Contains consecutive separators (empty layer)

    Examples:
        >>> parse_gate_layers(['h0', 'cnot0,1'])
        [['h0'], ['cnot0,1']]

        >>> parse_gate_layers(['h0', '-', 'cnot0,1'])
        [['h0'], ['cnot0,1']]

        >>> parse_gate_layers(['h0', 'h1', '-', 'cnot0,1', 'cnot2,3'])
        [['h0', 'h1'], ['cnot0,1', 'cnot2,3']]
    """
    # Handle empty input
    if not gates:
        return []

    # Check if there are any separators
    has_separators = LAYER_SEPARATOR in gates

    # If no separators, return each gate as its own layer (backward compatible)
    if not has_separators:
        return [[gate] for gate in gates]

    # Check for invalid separator positions
    if gates[0] == LAYER_SEPARATOR:
        raise ValueError(
            f"Gate list cannot start with separator '{LAYER_SEPARATOR}'. "
            f"First element must be a gate specification."
        )

    if gates[-1] == LAYER_SEPARATOR:
        raise ValueError(
            f"Gate list cannot end with separator '{LAYER_SEPARATOR}'. "
            f"Last element must be a gate specification."
        )

    # Parse layers
    layers: List[List[str]] = []
    current_layer: List[str] = []

    for gate in gates:
        if gate == LAYER_SEPARATOR:
            # Check for empty layer (consecutive separators)
            if not current_layer:
                raise ValueError(
                    f"Found empty layer (consecutive '{LAYER_SEPARATOR}' separators). "
                    f"Each layer must contain at least one gate."
                )

            # Save current layer and start new one
            layers.append(current_layer)
            current_layer = []
        else:
            # Add gate to current layer
            current_layer.append(gate)

    # Don't forget the last layer
    if current_layer:
        layers.append(current_layer)

    return layers
