"""Conversion results obey chemistry and conformer requirements across backends."""

from unittest.mock import Mock

import numpy as np
import pytest
from rdkit import Chem

from steamroll import SteamrollConversionError, SteamrollTopologyMismatchError, to_rdkit
from steamroll import steamroll as converter
from tests.test_steamroll import DATA_DIR, read_xyz


@pytest.mark.parametrize("strict", [False, True])
def test_metal_reference_is_enforced(monkeypatch: pytest.MonkeyPatch, strict: bool) -> None:
    """Metal inference must match reference bonds even when elements and charge agree."""
    numbers, coords, charge = read_xyz(DATA_DIR / "ferrocene.xyz")
    monkeypatch.setattr(
        converter, "_from_smiles_and_coords", Mock(side_effect=ValueError("No template mapping"))
    )
    with pytest.raises(SteamrollTopologyMismatchError):
        to_rdkit(
            numbers,
            coords,
            charge=charge,
            smiles="[Fe+2].[cH-]1cccc1.[cH-]1cccc1",
            fail_without_bond_order=strict,
        )


def test_multiple_metals_require_a_template() -> None:
    """Specialized inference rejects multiple centers instead of discarding an atom."""
    with pytest.raises(SteamrollConversionError):
        to_rdkit([30, 30], [[0, 0, 0], [10, 0, 0]], charge=4)
    mol = to_rdkit([30, 30], [[0, 0, 0], [10, 0, 0]], charge=4, smiles="[Zn+2].[Zn+2]")
    assert [atom.GetAtomicNum() for atom in mol.GetAtoms()] == [30, 30]
    assert Chem.GetFormalCharge(mol) == 4


@pytest.mark.parametrize("backend", ["_from_xyz", "_from_legacy", "get_tmc_mol"])
@pytest.mark.parametrize("defect", ["element", "charge", "coordinates", "nonfinite"])
def test_invalid_backend_results_are_rejected(
    monkeypatch: pytest.MonkeyPatch, backend: str, defect: str
) -> None:
    """Every chemistry backend rejects lost atoms, wrong charge, and invalid conformers."""
    mol = Chem.MolFromSmiles("[Cu+2]" if defect == "element" else "[Zn+2]")
    if defect == "charge":
        mol.GetAtomWithIdx(0).SetFormalCharge(1)
    if defect != "coordinates":
        conf = Chem.Conformer(1)
        conf.SetAtomPosition(0, [np.nan if defect == "nonfinite" else 1, 2, 3])
        mol.AddConformer(conf)
    monkeypatch.setattr(converter, "_from_xyz", Mock(side_effect=ValueError("No candidate")))
    monkeypatch.setattr(converter, "_from_legacy", Mock(return_value=None))
    monkeypatch.setattr(converter, backend, Mock(return_value=mol))
    if backend != "get_tmc_mol":
        monkeypatch.setattr(converter, "TRANSITION_METALS_NUM", [])
    with pytest.raises(SteamrollConversionError):
        to_rdkit([30], [[1, 2, 3]], charge=2)
