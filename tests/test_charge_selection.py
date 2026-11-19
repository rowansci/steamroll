"""Bounded inference prefers reasonable charges independently of input atom order."""

from subprocess import TimeoutExpired
from unittest.mock import Mock

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom

from steamroll import steamroll as converter
from steamroll import to_rdkit
from tests.test_steamroll import geometry_from_smiles


@pytest.mark.parametrize(
    ("seed", "order_seed", "remove_hs"),
    [
        pytest.param(1, None, False, id="original"),
        pytest.param(17, 0, False, id="permuted"),
        pytest.param(42, 19, True, id="permuted-without-hydrogens"),
    ],
)
def test_thiophene_charge_assignment(seed: int, order_seed: int | None, remove_hs: bool) -> None:
    """The reporter's original thiophene input and its permutations stay neutral."""
    ref = Chem.MolFromSmiles("s1cccc1C(=O)O")
    mol = Chem.AddHs(ref)
    assert rdDistGeom.EmbedMolecule(mol, randomSeed=seed) == 0
    numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    coords = mol.GetConformer().GetPositions()
    order = (
        list(range(len(numbers)))
        if order_seed is None
        else np.random.default_rng(order_seed).permutation(len(numbers)).tolist()
    )
    result = to_rdkit([numbers[i] for i in order], coords[order], remove_Hs=remove_hs)
    assert Chem.MolToSmiles(Chem.RemoveHs(result)) == Chem.MolToSmiles(ref)
    assert all(atom.GetFormalCharge() == 0 for atom in result.GetAtoms())
    kept = [i for i in order if not remove_hs or numbers[i] != 1]
    np.testing.assert_array_equal(result.GetConformer().GetPositions(), coords[kept])


def test_charge_refinement_preserves_indexed_connectivity(monkeypatch: pytest.MonkeyPatch) -> None:
    """A lower charge penalty cannot justify attaching bonds to different input atoms."""
    numbers, coords = geometry_from_smiles("s1cccc1C(=O)O")
    reference = Chem.AddHs(Chem.MolFromSmiles("s1cccc1C(=O)O"))
    order = list(range(len(numbers)))
    order[1], order[2] = order[2], order[1]
    candidate = Chem.RenumberAtoms(reference, order)
    conf = Chem.Conformer(len(numbers))
    for i, position in enumerate(coords):
        conf.SetAtomPosition(i, position)
    candidate.AddConformer(conf)
    legacy = Mock(return_value=candidate)
    monkeypatch.setattr(converter, "_from_legacy", legacy)
    result = to_rdkit(numbers, coords, remove_Hs=False)
    legacy.assert_called_once()
    np.testing.assert_array_equal(
        Chem.GetAdjacencyMatrix(result), Chem.GetAdjacencyMatrix(reference)
    )
    np.testing.assert_array_equal(result.GetConformer().GetPositions(), coords)


@pytest.mark.parametrize("smiles", ["[NH3+]CC(=O)[O-]", "C[n+]1ccsc1", "C[N+](C)(C)C"])
def test_required_charges_are_preserved(smiles: str) -> None:
    """Charge selection retains zwitterions and required cationic groups."""
    numbers, coords = geometry_from_smiles(smiles)
    ref = Chem.MolFromSmiles(smiles)
    result = to_rdkit(numbers, coords, charge=Chem.GetFormalCharge(ref))
    assert Chem.MolToSmiles(result) == Chem.MolToSmiles(ref)


def test_charge_refinement_timeout_preserves_valid_result(monkeypatch: pytest.MonkeyPatch) -> None:
    """A timeout during optional refinement preserves the assigned molecule."""
    smiles = "C[S-]"
    numbers, coords = geometry_from_smiles(smiles)
    legacy = Mock(side_effect=TimeoutExpired("steamroll._legacy", 5))
    monkeypatch.setattr(converter.subprocess, "run", legacy)
    result = to_rdkit(numbers, coords, charge=-1)
    assert Chem.MolToSmiles(result) == smiles
    legacy.assert_called_once()


def test_neutral_macrocycle_skips_charge_refinement(monkeypatch: pytest.MonkeyPatch) -> None:
    """Neutral sulfur macrocycles retain the bounded fast path."""
    numbers, coords = geometry_from_smiles("C1CCCCSCCCCCCCCSCCCC1")
    legacy = Mock(side_effect=AssertionError("Neutral macrocycle must not need refinement"))
    monkeypatch.setattr(converter, "_from_legacy", legacy)
    result = to_rdkit(numbers, coords)
    assert Chem.GetFormalCharge(result) == 0
    legacy.assert_not_called()
