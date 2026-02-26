"""Tests for the steamroll package."""

from pathlib import Path
from typing import Any

import pytest
from rdkit import Chem

from steamroll.steamroll import ATOMIC_NUMBERS, fragment, to_rdkit

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"


def parse_comment_line(comment: str) -> dict[str, Any]:
    """Parse the comment line of an XYZ file."""
    data: dict[str, Any] = {}
    for kv in comment.strip(";").split(";"):
        try:
            key, value = kv.split(":", 1)
            data[key.strip()] = value.strip()
        except ValueError:
            continue

    return data


def read_xyz(file: Path | str) -> tuple[list[int], list[list[float]], int]:
    """Read an XYZ file."""
    atomic_numbers = []
    coordinates = []
    with Path(file).open() as f:
        next(f)
        data = parse_comment_line(next(f))
        charge = int(data.get("charge", 0))

        for line in f:
            atom, x, y, z = line.split()
            if atom.isdigit():
                atomic_numbers.append(int(atom))
            else:
                atomic_numbers.append(ATOMIC_NUMBERS[atom])
            coordinates.append([float(x), float(y), float(z)])

    return atomic_numbers, coordinates, charge


def test_steamroll() -> None:
    """Basic test to make sure the package is working."""
    atomic_numbers = [1, 8, 1]
    coordinates = [[0, 0, 0], [0, 0, 1], [0, 1, 1]]

    rdkm = to_rdkit(atomic_numbers, coordinates)

    assert rdkm.GetNumAtoms() == 1


def test_no_remove_hydrogens() -> None:
    """Test to make sure hydrogens are removed."""
    atomic_numbers = [1, 8, 1]
    coordinates = [[0, 0, 0], [0, 0, 1], [0, 1, 1]]

    rdkm = to_rdkit(atomic_numbers, coordinates, remove_Hs=False)

    assert isinstance(rdkm, Chem.rdchem.Mol)
    assert rdkm.GetNumAtoms() == 3


def test_fragment() -> None:
    """Test to make sure multiple molecules are produced."""
    atomic_numbers = [1, 8, 1, 1, 8, 1]
    coordinates = [[0, 0, 0], [0, 0, 1], [0, 1, 1], [50, 0, 0], [50, 0, 1], [50, 1, 1]]

    rdkm = to_rdkit(atomic_numbers, coordinates)

    rdkm1, rdkm2 = fragment(rdkm)
    assert rdkm1.GetNumAtoms() == 1
    assert rdkm2.GetNumAtoms() == 1


@pytest.mark.parametrize("file", DATA_DIR.glob("*.xyz"))
def test_all_data(file: str) -> None:
    """Test to make sure all data files can be processed."""
    atomic_numbers, coordinates, charge = read_xyz(file)
    rdkm = to_rdkit(atomic_numbers, coordinates, charge=charge, remove_Hs=False)

    assert rdkm.GetNumAtoms() == len(atomic_numbers)
