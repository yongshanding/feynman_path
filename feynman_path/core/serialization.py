"""JSON serialization for Feynman path graphs with sympy support."""

import json
from typing import Any
import sympy


class SympyEncoder(json.JSONEncoder):
    """
    JSON encoder that handles sympy expressions and complex numbers.

    Sympy expressions are converted to strings like 'sqrt(2)/2'.
    Complex numbers are converted to strings like '(0.5+0.5j)'.
    """

    def default(self, obj: Any) -> Any:
        """
        Convert object to JSON-serializable form.

        Args:
            obj: Object to serialize

        Returns:
            JSON-serializable representation
        """
        # Handle sympy expressions
        if isinstance(obj, sympy.Basic):
            return str(obj)

        # Handle complex numbers
        if isinstance(obj, complex):
            # If imaginary part is zero, return just the real part
            if obj.imag == 0:
                return obj.real
            # Otherwise return string representation
            return str(obj)

        # Handle numpy types if present
        try:
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.complexfloating):
                if obj.imag == 0:
                    return float(obj.real)
                return str(obj)
        except ImportError:
            pass

        # Let the base class handle it
        return super().default(obj)


def to_json_string(data: dict, indent: int = 2) -> str:
    """
    Convert Feynman graph data to JSON string.

    Args:
        data: Dictionary containing Feynman path data
        indent: Number of spaces for indentation (default: 2)

    Returns:
        JSON string with proper formatting
    """
    return json.dumps(data, cls=SympyEncoder, indent=indent)


def to_json_file(data: dict, filename: str, indent: int = 2) -> None:
    """
    Save Feynman graph data to JSON file.

    Args:
        data: Dictionary containing Feynman path data
        filename: Path to output file
        indent: Number of spaces for indentation (default: 2)
    """
    with open(filename, 'w') as f:
        json.dump(data, f, cls=SympyEncoder, indent=indent)
