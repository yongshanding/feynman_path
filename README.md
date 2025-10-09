# Feynman Path Computation for Quantum Circuits

A computation tool for the Feynman Path Sum applied to quantum circuits with JSON output.
The [path integral formulation](https://en.wikipedia.org/wiki/Path_integral_formulation) is an interpretation of quantum mechanics that can aid in understanding superposition and interference.

This tool computes the Feynman path representation of quantum circuits and outputs the result in JSON format, suitable for further processing or visualization.

## Key Features

- **Sparse State Representation**: Efficient computation for large qubit systems by storing only non-zero amplitude states
- **Generic Gate System**: Supports arbitrary unitary matrices, not limited to predefined gates
- **Symbolic Computation**: Uses sympy for exact symbolic amplitude representations
- **JSON Output**: Structured data format suitable for further processing or custom visualization
- **Production-Ready**: Clean architecture with comprehensive test coverage (59+ tests)

# Install

feynman\_path is available on PyPI:

```bash
python3 -m pip install feynman_path
```

## Dependencies

The tool requires only Python dependencies:
- `numpy>=1.20`: For matrix operations and quantum gate computations
- `sympy>=1.7`: For symbolic amplitude representations

These will be automatically installed when you install the package.


# Usage

This package provides a command line tool to compute Feynman path representations and output JSON.

## Command Line Interface

### Basic Usage

Output JSON to stdout:
```bash
feynman_path 2 h0 cnot0,1 z1 h0 h1 cnot1,0 h1
```

Save JSON to a file:
```bash
feynman_path 2 h0 cnot0,1 z1 h0 h1 cnot1,0 h1 --output result.json
```

### Examples

Simple 2-qubit circuit:
```bash
feynman_path 2 h0 cnot0,1
```

3-qubit circuit with various gates:
```bash
feynman_path 3 h0 cnot0,1 cnot1,2 h2
```

### Gate Specifications

Single-qubit gates: `h0`, `x1`, `z2`, `s0` (gate name + qubit index)
- `h`: Hadamard gate
- `x`: Pauli-X (NOT) gate
- `z`: Pauli-Z gate
- `s`: S gate (phase gate)

Two-qubit gates: `cnot0,1`, `conot1,2` (gate name + control,target)
- `cnot`: Controlled-NOT gate
- `conot`: Controlled-NOT with phase flip

### Command Line Options
```
$ feynman_path -h
usage: feynman_path [-h] [-o OUTPUT] [--indent INDENT] [--initial-state INITIAL_STATE]
                    n_qubits gates [gates ...]

Generate Feynman path representation for quantum circuits (JSON output only).

positional arguments:
  n_qubits              Number of qubits in the quantum circuit
  gates                 List of gates to apply (e.g., h0 cnot0,1 z1)

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output file path (default: print to stdout)
  --indent INDENT       JSON indentation level (default: 2)
  --initial-state INITIAL_STATE
                        Initial state as bitstring (default: 00...0)
```

### JSON Output Format

The output is a JSON object with the following structure:
```json
{
  "type": "feynmanpath",
  "cols": [
    {
      "state": {
        "amp": <amplitude>,
        "next": {
          "next_state_1": <transition_amplitude_1>,
          "next_state_2": <transition_amplitude_2>
        }
      }
    }
  ]
}
```

Each column represents a timestep (gate application), showing:
- Current states and their amplitudes
- Transitions to next states with transition amplitudes

## Python Package

feynman\_path also provides a Python 3 package as an alternative to the command line tool.

```python
from feynman_path.core import FeynmanGraph, parse_gate_spec

# Create a 2-qubit Feynman graph
graph = FeynmanGraph(n_qubits=2)

# Apply gates
gates = ['h0', 'cnot0,1', 'z1', 'h0', 'h1', 'cnot1,0', 'h1']
for gate_str in gates:
    gate = parse_gate_spec(gate_str, n_qubits=2)
    graph.apply_gate(gate)

# Get JSON output
json_data = graph.to_json()

# Get final quantum state
final_state = graph.get_final_state()
print(final_state)  # {'10': 0.9999999999999996}
```

### Saving to File

```python
from feynman_path.core import to_json_file, to_json_string

# Save to file
to_json_file(json_data, 'output.json', indent=2)

# Or get as string
json_str = to_json_string(json_data, indent=2)
print(json_str)
```

### Custom Gate Matrices

```python
from feynman_path.core import Gate
import numpy as np

# Define a custom unitary gate
custom_matrix = np.array([
    [1, 0],
    [0, 1j]
])

# Create gate for qubit 0
custom_gate = Gate(
    name='CustomPhase',
    matrix=custom_matrix,
    qubits=(0,),
    n_qubits=2
)

graph.apply_gate(custom_gate)
```

# Circuit Examples

### Creating a Bell Pair (Entanglement)

The [CNOT gate](https://en.wikipedia.org/wiki/Controlled_NOT_gate) can be used to entangle two qubits, creating a [Bell pair](https://en.wikipedia.org/wiki/Bell_state):

```bash
feynman_path 2 h0 cnot0,1 --output bell_pair.json
```

This creates the entangled state |00⟩+|11⟩ (up to normalization). The JSON output shows all paths through the computation with their amplitudes.

### Quantum Interference Circuit

This circuit demonstrates quantum interference where multiple paths cancel:

```bash
feynman_path 2 h0 cnot0,1 z1 h0 h1 cnot1,0 h1 --output interference.json
```

The final state shows the result after interference between different computational paths.

### Multi-Qubit Circuits

Compute paths for a 3-qubit circuit:

```bash
feynman_path 3 h0 cnot0,1 z1 cnot1,2 h0 h1 cnot1,0 h1 --output three_qubit.json
```

The sparse representation efficiently handles larger qubit systems by only tracking non-zero amplitude states.

## Qubit Ordering Convention

This package uses the convention where `bitstring[i] = qubit_i`. For example:
- `'011'` means q0=0, q1=1, q2=1
- `'100'` means q0=1, q1=0, q2=0

This is consistent throughout the codebase, JSON output, and internal state representations.
