"""Utility functions for parsing element names from TIFF filenames."""

import re
from pathlib import Path
from typing import Optional


def extract_element_from_filename(filepath: str) -> str:
    """
    Extract chemical element symbol from TIFF filename.

    Handles various filename patterns:
    - detsum_Ca_K.tiff → Ca
    - detsum_Ce_L.tiff → Ce
    - Ca_K.tiff → Ca
    - calcium.tiff → Ca
    - Fe.tiff → Fe

    Args:
        filepath: Path to the TIFF file

    Returns:
        Element symbol (e.g., "Ca", "Fe", "Ce")
    """
    filename = Path(filepath).stem  # Remove extension

    # Pattern 1: detsum_Element_Line format (detsum_Ca_K, detsum_Ce_L)
    # Look for: detsum_ELEMENT_ specifically
    match = re.search(r'detsum_([A-Z][a-z]?)_', filename, re.IGNORECASE)
    if match:
        return match.group(1)

    # Pattern 2: Element_Line format at start (Ca_K, Ce_L, Fe_K, S_K)
    # Look for: ELEMENT_ at the very start of filename
    match = re.search(r'^([A-Z][a-z]?)_', filename)
    if match:
        return match.group(1)

    # Pattern 3: Just element with underscore (Ca_, Fe_)
    match = re.search(r'(?:^|_)([A-Z][a-z]?)_', filename)
    if match:
        return match.group(1)

    # Pattern 4: Element at the start (Ca, Fe, Calcium)
    match = re.search(r'^([A-Z][a-z]?)', filename)
    if match:
        return match.group(1)

    # Pattern 5: Try to find any 1-2 letter element symbol in the filename
    # Common elements in X-ray fluorescence
    common_elements = [
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
        'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
        'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
        'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
        'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
        'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm'
    ]

    # Look for element symbols in the filename
    for element in common_elements:
        if re.search(rf'\b{element}\b', filename, re.IGNORECASE):
            return element

    # Fallback: Return first part of filename (capitalize first letter)
    parts = re.split(r'[_\-\s]', filename)
    if parts:
        first_part = parts[0]
        # Capitalize first letter, lowercase rest (unless it's all caps)
        if len(first_part) <= 2 and first_part.isupper():
            return first_part
        return first_part[:1].upper() + first_part[1:].lower() if len(first_part) > 1 else first_part.upper()

    # Ultimate fallback
    return "Unknown"


def validate_element_symbol(symbol: str) -> bool:
    """
    Validate if a string is a valid chemical element symbol.

    Args:
        symbol: Potential element symbol

    Returns:
        True if valid element symbol
    """
    # List of all valid element symbols
    valid_elements = {
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
        'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca',
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
        'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr',
        'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
        'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
        'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
        'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
        'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
        'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
        'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
        'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
    }

    return symbol in valid_elements


# Test cases
if __name__ == "__main__":
    test_cases = [
        ("detsum_Ca_K.tiff", "Ca"),
        ("detsum_Ce_L.tiff", "Ce"),
        ("Ca_K.tiff", "Ca"),
        ("Fe.tiff", "Fe"),
        ("calcium.tiff", "Ca"),
        ("iron_map.tiff", "Fe"),
        ("S_K_alpha.tiff", "S"),
        ("detsum_Pb_M.tiff", "Pb"),
    ]

    print("Testing element extraction:")
    for filename, expected in test_cases:
        result = extract_element_from_filename(filename)
        status = "✓" if result == expected else "✗"
        print(f"{status} {filename:30s} → {result:3s} (expected: {expected})")
