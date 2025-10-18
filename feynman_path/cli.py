"""Command-line interface for Feynman path generation."""

import argparse
import sys
from .core import FeynmanGraph, parse_gate_spec, to_json_string, to_json_file, parse_gate_layers


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Generate Feynman path representation for quantum circuits (JSON output only).',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Output JSON to stdout
  feynman_path 2 h0 cnot0,1 z1 h0 h1 cnot1,0 h1

  # Save to file
  feynman_path 2 h0 cnot0,1 --output result.json

  # Rotation gates with angles
  feynman_path 2 rx0,pi/4 ry1,1.5708 cnot0,1

  # Toffoli and multi-controlled gates
  feynman_path 4 h0 h1 h2 m3cnot0,1,2,3

Gate specifications:
  Single-qubit gates: h0, x1, y2, z0, s1, sdag0, t1, tdag2
  Rotation gates: rx0,pi/4, ry1,1.5708, rz2,pi/2 (qubit,angle)
  Two-qubit gates: cnot0,1, conot1,2 (control,target)
  Multi-qubit gates: toffoli0,1,2, m3cnot0,1,2,3 (controls...,target)

Available gates:
  - Basic: H, X, Y, Z, S, Sdag, T, Tdag
  - Rotations: Rx, Ry, Rz (angles: numeric, pi, or expressions like pi/4)
  - Controlled: CNOT, CONOT, Toffoli, M[k]CNOT (k=1 to 15)

Layer-based mode:
  Use '-' to separate gates into layers. Each layer produces one column in the output
  instead of one column per gate. Gates within a layer are applied sequentially.

  Example: feynman_path 4 h0 h1 h2 h3 - cnot0,1 cnot2,3 - cnot1,2
           This creates 4 columns: initial + 3 layers

  Without '-', each gate creates its own column (original behavior).
        """
    )

    parser.add_argument(
        'n_qubits',
        type=int,
        help='Number of qubits in the quantum circuit'
    )

    parser.add_argument(
        'gates',
        nargs='+',
        help='List of gates to apply (e.g., h0 cnot0,1 z1). Use - to separate layers'
    )

    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output file path (default: print to stdout)'
    )

    parser.add_argument(
        '--indent',
        type=int,
        default=2,
        help='JSON indentation level (default: 2)'
    )

    parser.add_argument(
        '--initial-state',
        type=str,
        default=None,
        help='Initial state as bitstring (default: 00...0)'
    )

    args = parser.parse_args()

    try:
        # Create Feynman graph
        graph = FeynmanGraph(
            n_qubits=args.n_qubits,
            initial_state=args.initial_state
        )

        # Parse gate layers (handles both layer mode and regular mode)
        try:
            layers = parse_gate_layers(args.gates)
        except ValueError as e:
            print(f"Error parsing gate layers: {e}", file=sys.stderr)
            sys.exit(1)

        # Apply each layer
        for layer_gates in layers:
            try:
                # Parse gate specifications
                gates = [parse_gate_spec(g, n_qubits=args.n_qubits) for g in layer_gates]

                # Apply as layer if multiple gates, otherwise use apply_gate
                if len(gates) == 1:
                    graph.apply_gate(gates[0])
                else:
                    graph.apply_layer(gates)

            except (ValueError, KeyError) as e:
                print(f"Error processing gates {layer_gates}: {e}", file=sys.stderr)
                sys.exit(1)

        # Get JSON output
        json_data = graph.to_json()

        # Output
        if args.output:
            to_json_file(json_data, args.output, indent=args.indent)
            print(f"Feynman path saved to {args.output}", file=sys.stderr)
        else:
            print(to_json_string(json_data, indent=args.indent))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
