"""Tests for the steamroll package."""

from rdkit import Chem

from steamroll.steamroll import fragment, to_rdkit


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


def test_fragement() -> None:
    """Test to make sure multiple molecules are produced."""
    atomic_numbers = [1, 8, 1, 1, 8, 1]
    coordinates = [[0, 0, 0], [0, 0, 1], [0, 1, 1], [50, 0, 0], [50, 0, 1], [50, 1, 1]]

    rdkm = to_rdkit(atomic_numbers, coordinates)

    rdkm1, rdkm2 = fragment(rdkm)
    assert rdkm1.GetNumAtoms() == 1
    assert rdkm2.GetNumAtoms() == 1
