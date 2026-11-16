"""Regression tests for transition-metal atom and coordinate correspondence."""

import numpy as np
import pytest
from rdkit import Chem

from steamroll import to_rdkit


@pytest.mark.parametrize("remove_hs", [False, True])
def test_zinc_chloride_preserves_input_order(remove_hs: bool) -> None:
    """Zinc chloride retains input atom order and supplied coordinates."""
    numbers = [30, 17, 17]
    coords = [[0.0, 0.0, 0.0], [2.2, 0.0, 0.0], [-2.2, 0.0, 0.0]]
    mol = to_rdkit(numbers, coords, remove_Hs=remove_hs)

    assert [atom.GetAtomicNum() for atom in mol.GetAtoms()] == numbers
    np.testing.assert_allclose(mol.GetConformer().GetPositions(), coords, rtol=0, atol=1e-12)
    assert Chem.GetFormalCharge(mol) == 0
    Chem.SanitizeMol(mol)


@pytest.mark.parametrize("remove_hs", [False, True])
@pytest.mark.parametrize("order", [(0, 1, 2, 3, 4), (2, 1, 3, 0, 4)])
def test_zinc_ammonia_preserves_supplied_hydrogen_coordinates(
    remove_hs: bool, order: tuple[int, ...]
) -> None:
    """Shuffled zinc ammonia preserves supplied positions through hydrogen handling."""
    numbers = [7, 30, 1, 1, 1]
    coords = [
        [2.1, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [2.7, 0.9, 0.0],
        [2.7, -0.45, 0.78],
        [2.7, -0.45, -0.78],
    ]
    numbers = [numbers[i] for i in order]
    coords = [coords[i] for i in order]
    mol = to_rdkit(numbers, coords, charge=2, remove_Hs=remove_hs)
    kept = [i for i, number in enumerate(numbers) if not remove_hs or number != 1]

    assert [atom.GetAtomicNum() for atom in mol.GetAtoms()] == [numbers[i] for i in kept]
    np.testing.assert_allclose(
        mol.GetConformer().GetPositions(), [coords[i] for i in kept], rtol=0, atol=1e-12
    )
    assert Chem.GetFormalCharge(mol) == 2
    Chem.SanitizeMol(mol)
