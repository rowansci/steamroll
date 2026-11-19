"""Topology templates reject incompatible input before inference."""

from unittest.mock import Mock

import numpy as np
import pytest
from rdkit import Chem

from steamroll import SteamrollTopologyMismatchError, to_rdkit
from steamroll import steamroll as converter
from tests.test_steamroll import DATA_DIR, geometry_from_smiles, read_xyz

_BZU = "CCN[C@H]1CN(CCOCC)S(=O)(=O)c2sc(cc12)S(N)(=O)=O"


@pytest.mark.parametrize(
    ("numbers", "coords", "charge", "smiles", "error"),
    [
        ([6], [[0, 0, 0]], 0, "invalid!", ValueError),
        ([30], [[0, 0, 0]], 2, "[Cu+2]", SteamrollTopologyMismatchError),
        ([8], [[0, 0, 0]], 1, "O", SteamrollTopologyMismatchError),
        ([6, 1], [[0, 0, 0], [1, 0, 0]], 0, "C", SteamrollTopologyMismatchError),
        ([6], [[0, 0, 0]], 0, "[2H]C", SteamrollTopologyMismatchError),
    ],
)
def test_incompatible_template_fails_before_inference(
    monkeypatch: pytest.MonkeyPatch,
    numbers: list[int],
    coords: list[list[float]],
    charge: int,
    smiles: str,
    error: type[Exception],
) -> None:
    """Invalid templates, missing elements, partial Hs, and charge conflicts fail promptly."""
    for backend in ("_from_smiles_and_coords", "_from_xyz", "_from_legacy", "get_tmc_mol"):
        monkeypatch.setattr(converter, backend, Mock(side_effect=AssertionError("No inference")))
    with pytest.raises(error):
        to_rdkit(numbers, coords, charge=charge, smiles=smiles)


def test_incomplete_bzu_fails_before_inference(monkeypatch: pytest.MonkeyPatch) -> None:
    """The public PDB example lacks a heavy atom required by the BZU template."""
    numbers, coords, charge = read_xyz(DATA_DIR / "invalid" / "bzu_incomplete.xyz")
    for backend in ("_from_xyz", "_from_legacy"):
        monkeypatch.setattr(converter, backend, Mock(side_effect=AssertionError("No inference")))
    with pytest.raises(SteamrollTopologyMismatchError):
        to_rdkit(numbers, coords, charge=charge, smiles=_BZU)


@pytest.mark.parametrize("hydrogens", [False, True])
def test_complete_bzu_accepts_template(hydrogens: bool) -> None:
    """Complete BZU with all or no hydrogen coordinates recovers reference chemistry."""
    numbers, coords = geometry_from_smiles(_BZU)
    kept = [i for i, number in enumerate(numbers) if hydrogens or number != 1]
    result = to_rdkit(
        [numbers[i] for i in kept], [coords[i] for i in kept], smiles=_BZU, remove_Hs=False
    )
    assert Chem.MolToSmiles(Chem.RemoveHs(result)) == Chem.MolToSmiles(Chem.MolFromSmiles(_BZU))
    np.testing.assert_array_equal(result.GetConformer().GetPositions(), [coords[i] for i in kept])
