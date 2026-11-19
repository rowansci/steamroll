"""Geometry-only conversion returns actual bonds without inventing molecular chemistry."""

from collections.abc import Callable
from unittest.mock import Mock

import numpy as np
import pytest
from numpy.typing import ArrayLike
from rdkit import Chem

from steamroll import (
    SteamrollConversionError,
    SteamrollTopologyMismatchError,
    to_connectivity,
    to_rdkit,
)
from steamroll import steamroll as converter
from tests.test_steamroll import DATA_DIR, geometry_from_smiles, read_xyz

_ETHANOL_NUMBERS = [6, 6, 8]
_ETHANOL_COORDS = [[0, 0, 0], [1.54, 0, 0], [2.4, 1.1, 0]]


def test_direct_connectivity_skips_chemistry_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    """Heavy-atom ethanol acquires two single bonds without bond-order inference."""
    for backend in ("_from_xyz", "_from_legacy", "get_tmc_mol"):
        monkeypatch.setattr(converter, backend, Mock(side_effect=AssertionError("No chemistry")))
    mol = to_connectivity(iter(_ETHANOL_NUMBERS), _ETHANOL_COORDS)
    assert {tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))) for b in mol.GetBonds()} == {
        (0, 1),
        (1, 2),
    }
    assert all(b.GetBondType() == Chem.BondType.SINGLE for b in mol.GetBonds())
    assert [atom.GetTotalNumHs() for atom in mol.GetAtoms()] == [0, 0, 0]
    assert not mol.GetBoolProp("steamroll_bond_orders_assigned")
    np.testing.assert_array_equal(mol.GetConformer().GetPositions(), _ETHANOL_COORDS)


def test_permissive_fallback_contains_bonds(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only permissive conversion returns an explicitly marked connectivity graph."""
    monkeypatch.setattr(converter, "_from_xyz", Mock(side_effect=ValueError("No bond orders")))
    monkeypatch.setattr(converter, "_from_legacy", Mock(return_value=None))
    with pytest.raises(SteamrollConversionError):
        to_rdkit(_ETHANOL_NUMBERS, _ETHANOL_COORDS)
    mol = to_rdkit(_ETHANOL_NUMBERS, _ETHANOL_COORDS, fail_without_bond_order=False)
    assert mol.GetNumBonds() == 2
    assert not mol.GetBoolProp("steamroll_bond_orders_assigned")


def test_incomplete_pdb_connectivity() -> None:
    """The incomplete BZU geometry returns bonds without claiming reference chemistry."""
    numbers, coords, _ = read_xyz(DATA_DIR / "invalid" / "bzu_incomplete.xyz")
    mol = to_connectivity(numbers, coords)
    assert mol.GetNumAtoms() == 22
    assert mol.GetNumBonds() == 21
    assert mol.GetBondBetweenAtoms(8, 9) is not None
    assert [atom.GetAtomicNum() for atom in mol.GetAtoms()] == numbers
    assert not mol.GetBoolProp("steamroll_bond_orders_assigned")
    np.testing.assert_array_equal(mol.GetConformer().GetPositions(), coords)


@pytest.mark.parametrize("remove_hs", [False, True])
def test_connectivity_hydrogen_policy(remove_hs: bool) -> None:
    """Removing supplied water hydrogens retains the oxygen's input coordinates."""
    numbers = [1, 8, 1]
    coords = [[0, 0, 0], [0, 0, 1], [0, 1, 1]]
    mol = to_connectivity(numbers, coords, remove_Hs=remove_hs)
    kept = [1] if remove_hs else [0, 1, 2]
    assert [atom.GetAtomicNum() for atom in mol.GetAtoms()] == [numbers[i] for i in kept]
    np.testing.assert_array_equal(mol.GetConformer().GetPositions(), [coords[i] for i in kept])


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize(
    ("geometry", "reference"),
    [
        ("[NH3+][C@@H](C)C(=O)[O-]", "[NH3+][C@H](C)C(=O)[O-]"),
        ("C[N@@+](CC)(CCC)CCCC", "C[N@+](CC)(CCC)CCCC"),
    ],
)
def test_charged_stereo_conflict_rejects_unassigned_fallback(
    geometry: str, reference: str, strict: bool
) -> None:
    """Charged stereo conflicts raise the domain exception under both failure policies."""
    numbers, coords = geometry_from_smiles(geometry)
    charge = Chem.GetFormalCharge(Chem.MolFromSmiles(geometry))
    with pytest.raises(SteamrollTopologyMismatchError):
        to_rdkit(numbers, coords, charge=charge, smiles=reference, fail_without_bond_order=strict)


@pytest.mark.parametrize(
    ("numbers", "coords", "charge", "smiles"),
    [
        ([1, 8, 1], [[0, 0, 0], [0, 0, 1], [0, 1, 1]], 0, None),
        ([30], [[1, 2, 3]], 2, None),
        (_ETHANOL_NUMBERS, _ETHANOL_COORDS, 0, "CCO"),
    ],
)
def test_assigned_chemistry_is_identified(
    numbers: list[int], coords: list[list[float]], charge: int, smiles: str | None
) -> None:
    """Organic, metal, and template conversion identify results with assigned chemistry."""
    mol = to_rdkit(numbers, coords, charge=charge, smiles=smiles)
    assert mol.GetBoolProp("steamroll_bond_orders_assigned")


@pytest.mark.parametrize("convert", [to_rdkit, to_connectivity])
@pytest.mark.parametrize(
    ("numbers", "coords"),
    [
        ([], np.empty((0, 3))),
        ([6], [[0, 0]]),
        ([6, 8], [[0, 0, 0]]),
        ([0], [[0, 0, 0]]),
        ([6.5], [[0, 0, 0]]),
        ([6], [[np.nan, 0, 0]]),
    ],
)
def test_invalid_geometry_is_rejected(
    convert: Callable[[list[int], ArrayLike], Chem.rdchem.Mol],
    numbers: list[int],
    coords: ArrayLike,
) -> None:
    """Both conversion entry points enforce the same geometry requirements."""
    with pytest.raises(ValueError, match=r"Coordinates|Atomic numbers"):
        convert(numbers, coords)
