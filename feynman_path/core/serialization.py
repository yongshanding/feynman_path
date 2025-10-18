"""JSON serialization for Feynman path graphs with sympy support."""

import json
from typing import Any, Dict, List
import sympy


class SympyEncoder(json.JSONEncoder):
    """
    JSON encoder that handles sympy expressions and complex numbers.

    Sympy expressions are converted to strings like 'sqrt(2)/2'.
    Complex numbers are converted to objects like { re: 0.5, im: 0.5 }.
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
            # Otherwise return as {re, im} object
            return {'re': obj.real, 'im': obj.imag}

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
                return {'re': float(obj.real), 'im': float(obj.imag)}
        except ImportError:
            pass

        # Let the base class handle it
        return super().default(obj)


def _format_value(value: Any, indent_level: int, indent: int) -> str:
    """
    Format a value for custom JSON output.

    Args:
        value: Value to format
        indent_level: Current indentation level
        indent: Number of spaces per indent level

    Returns:
        Formatted string representation
    """
    if value is None:
        return 'null'
    elif isinstance(value, bool):
        return 'true' if value else 'false'
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, str):
        # Escape special characters and wrap in quotes
        return json.dumps(value)
    elif isinstance(value, sympy.Basic):
        return json.dumps(str(value))
    elif isinstance(value, complex):
        if value.imag == 0:
            return str(value.real)
        # Return as {re, im} object
        return _format_dict({'re': value.real, 'im': value.imag}, indent_level, indent)
    elif isinstance(value, dict):
        return _format_dict(value, indent_level, indent)
    elif isinstance(value, list):
        return _format_list(value, indent_level, indent)
    else:
        return json.dumps(value, cls=SympyEncoder)


def _format_dict(data: Dict, indent_level: int, indent: int) -> str:
    """
    Format a dictionary with unquoted keys for specific fields.

    Args:
        data: Dictionary to format
        indent_level: Current indentation level
        indent: Number of spaces per indent level

    Returns:
        Formatted string representation
    """
    if not data:
        return '{}'

    # Keys that should not have quotes
    unquoted_keys = {'type', 'cols', 'amp', 'next', 're', 'im'}

    spaces = ' ' * (indent_level * indent)
    inner_spaces = ' ' * ((indent_level + 1) * indent)

    lines = ['{']
    items = list(data.items())

    for i, (key, value) in enumerate(items):
        # Determine if key should be quoted
        if key in unquoted_keys:
            key_str = key
        else:
            key_str = json.dumps(key)

        value_str = _format_value(value, indent_level + 1, indent)

        # Add comma except for last item
        comma = ',' if i < len(items) - 1 else ''
        lines.append(f'{inner_spaces}{key_str}: {value_str}{comma}')

    lines.append(f'{spaces}}}')
    return '\n'.join(lines)


def _format_list(data: List, indent_level: int, indent: int) -> str:
    """
    Format a list.

    Args:
        data: List to format
        indent_level: Current indentation level
        indent: Number of spaces per indent level

    Returns:
        Formatted string representation
    """
    if not data:
        return '[]'

    spaces = ' ' * (indent_level * indent)
    inner_spaces = ' ' * ((indent_level + 1) * indent)

    lines = ['[']

    for i, item in enumerate(data):
        item_str = _format_value(item, indent_level + 1, indent)
        comma = ',' if i < len(data) - 1 else ''
        lines.append(f'{inner_spaces}{item_str}{comma}')

    lines.append(f'{spaces}]')
    return '\n'.join(lines)


def to_json_string(data: dict, indent: int = 2) -> str:
    """
    Convert Feynman graph data to JSON string with unquoted structural keys.

    Args:
        data: Dictionary containing Feynman path data
        indent: Number of spaces for indentation (default: 2)

    Returns:
        JSON-like string with unquoted keys for type, cols, amp, next
    """
    return _format_dict(data, indent_level=0, indent=indent)


def to_json_file(data: dict, filename: str, indent: int = 2) -> None:
    """
    Save Feynman graph data to JSON file with unquoted structural keys.

    Args:
        data: Dictionary containing Feynman path data
        filename: Path to output file
        indent: Number of spaces for indentation (default: 2)
    """
    with open(filename, 'w') as f:
        f.write(to_json_string(data, indent=indent))
