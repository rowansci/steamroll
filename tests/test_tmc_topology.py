"""Regression tests for specialized transition-metal topology validation."""

from unittest.mock import Mock

import pytest
from rdkit import Chem

from steamroll import SteamrollTopologyMismatchError, to_rdkit
from steamroll import steamroll as converter


@pytest.mark.parametrize("fail_without_bond_order", [False, True])
@pytest.mark.parametrize("remove_hs", [False, True])
def test_tmc_rejects_conflicting_smiles(fail_without_bond_order: bool, remove_hs: bool) -> None:
    """Reject copper reference for zinc input even with permissive fallback."""
    with pytest.raises(SteamrollTopologyMismatchError, match="SMILES"):
        to_rdkit(
            [30],
            [[1.0, 2.0, 3.0]],
            charge=2,
            smiles="[Cu+2]",
            remove_Hs=remove_hs,
            fail_without_bond_order=fail_without_bond_order,
        )


@pytest.mark.parametrize("smiles", [None, "[Zn+2]"])
@pytest.mark.parametrize("remove_hs", [False, True])
def test_tmc_accepts_matching_or_absent_reference(
    monkeypatch: pytest.MonkeyPatch, smiles: str | None, remove_hs: bool
) -> None:
    """Validate actual specialized result after template conversion fails."""
    template = Mock(side_effect=ValueError("Use specialized conversion"))
    monkeypatch.setattr(converter, "_from_smiles_and_coords", template)
    mol = to_rdkit([30], [[1.0, 2.0, 3.0]], charge=2, smiles=smiles, remove_Hs=remove_hs)
    assert Chem.MolToSmiles(mol) == "[Zn+2]"
    assert Chem.GetFormalCharge(mol) == 2
