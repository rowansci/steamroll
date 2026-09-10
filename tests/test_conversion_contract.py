"""Regression coverage for bounded organic inference and stereo preservation."""

from unittest.mock import Mock

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers

from steamroll import SteamrollConversionError, SteamrollTopologyMismatchError, to_rdkit
from steamroll import steamroll as converter
from tests.test_steamroll import DATA_DIR, geometry_from_smiles, read_xyz

_MACROCYCLE_SMILES = "C1CCCCSCCCCCCCCSCCCC1"


def _isomeric(mol: Chem.Mol) -> str:
    return Chem.MolToSmiles(Chem.RemoveHs(mol))


@pytest.mark.parametrize("guided", [False, True])
@pytest.mark.parametrize("remove_hs", [False, True])
def test_macrocycle_fast_path(
    monkeypatch: pytest.MonkeyPatch, guided: bool, remove_hs: bool
) -> None:
    """Synthetic sulfur macrocycle converts without reaching legacy inference."""
    legacy = Mock(side_effect=AssertionError("Macrocycle must use the fast path"))
    monkeypatch.setattr(converter, "_from_legacy", legacy)
    numbers, coords = geometry_from_smiles(_MACROCYCLE_SMILES)
    charge = 0
    mol = to_rdkit(
        numbers,
        coords,
        charge=charge,
        remove_Hs=remove_hs,
        smiles=_MACROCYCLE_SMILES if guided else None,
    )
    assert _isomeric(mol) == _isomeric(Chem.MolFromSmiles(_MACROCYCLE_SMILES))
    assert Chem.GetFormalCharge(mol) == 0
    legacy.assert_not_called()
    if not remove_hs:
        np.testing.assert_allclose(mol.GetConformer().GetPositions(), coords, atol=1e-12)


@pytest.mark.parametrize(
    "smiles",
    [
        "CSC",
        "CSSC",
        "CCS",
        "c1ccsc1",
        "C[S@](=O)CC",
        "CS(=O)(=O)C",
        "CS(=O)(=O)N",
        "CS(=O)(=O)[O-]",
        "C[S+](C)C",
        "CC[S-]",
        "[NH3+]CCS(=O)(=O)[O-]",
        "[O-]S(=O)(=O)[O-]",
        "C[n+]1ccsc1",
    ],
)
def test_sulfur_round_trip(smiles: str) -> None:
    """Sulfur functional groups and charges recover through bounded inference."""
    numbers, coords = geometry_from_smiles(smiles)
    ref = Chem.MolFromSmiles(smiles)
    mol = to_rdkit(numbers, coords, charge=Chem.GetFormalCharge(ref), remove_Hs=False)
    assert _isomeric(mol) == _isomeric(ref)
    assert Chem.GetFormalCharge(mol) == Chem.GetFormalCharge(ref)
    np.testing.assert_allclose(mol.GetConformer().GetPositions(), coords, atol=1e-12)


@pytest.mark.parametrize("smiles", [None, "C[S@](=O)CC", "C[S@+]([O-])CC"])
def test_sulfoxide_representation_and_mmff(smiles: str | None) -> None:
    """Equivalent sulfoxide inputs use identical graph and MMFF parameters."""
    numbers, coords = geometry_from_smiles("C[S@](=O)CC")
    ref = Chem.AddHs(Chem.MolFromSmiles("C[S@](=O)CC"))
    conf = Chem.Conformer(len(numbers))
    for i, position in enumerate(coords):
        conf.SetAtomPosition(i, position)
    ref.AddConformer(conf)
    mol = to_rdkit(numbers, coords, smiles=smiles, remove_Hs=False)
    assert _isomeric(mol) == _isomeric(ref)
    props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol)
    ref_props = rdForceFieldHelpers.MMFFGetMoleculeProperties(ref)
    assert props is not None
    assert ref_props is not None
    assert [props.GetMMFFAtomType(i) for i in range(len(numbers))] == [
        ref_props.GetMMFFAtomType(i) for i in range(len(numbers))
    ]
    assert rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, props).CalcEnergy() == pytest.approx(
        rdForceFieldHelpers.MMFFGetMoleculeForceField(ref, ref_props).CalcEnergy(), abs=1e-8
    )


@pytest.mark.parametrize("smiles", ["N[C@@H](C)C(=O)O", "F/C=C/Cl", "C[S@](=O)CC"])
@pytest.mark.parametrize("remove_hs", [False, True])
def test_shuffled_stereo(smiles: str, remove_hs: bool) -> None:
    """Atom reordering and hydrogen removal preserve specified stereo."""
    numbers, coords = geometry_from_smiles(smiles)
    order = np.random.default_rng(42).permutation(len(numbers)).tolist()
    mol = to_rdkit(
        [numbers[i] for i in order],
        [coords[i] for i in order],
        smiles=smiles,
        remove_Hs=remove_hs,
    )
    assert _isomeric(mol) == _isomeric(Chem.MolFromSmiles(smiles))


@pytest.mark.parametrize(
    ("geometry", "reference"),
    [("N[C@@H](C)C(=O)O", "N[C@H](C)C(=O)O"), ("F/C=C/Cl", "F/C=C\\Cl")],
)
def test_stereo_conflict_raises(geometry: str, reference: str) -> None:
    """Conflicting specified stereochemistry fails even with permissive fallback."""
    numbers, coords = geometry_from_smiles(geometry)
    with pytest.raises(SteamrollTopologyMismatchError):
        to_rdkit(numbers, coords, smiles=reference, fail_without_bond_order=False)


def test_unspecified_stereo_is_coordinate_derived() -> None:
    """Unspecified reference centers acquire geometry-derived stereochemistry."""
    numbers, coords = geometry_from_smiles("N[C@@H](C)C(=O)O")
    mol = to_rdkit(numbers, coords, smiles="NC(C)C(=O)O")
    assert _isomeric(mol) == _isomeric(Chem.MolFromSmiles("N[C@@H](C)C(=O)O"))


@pytest.mark.parametrize("smiles", ["[13CH3:7][C@@H](F)Cl", "[2H][C@@](F)(Cl)Br"])
def test_template_isotope_and_atom_map(smiles: str) -> None:
    """Template isotope and mapping labels survive coordinate assignment."""
    numbers, coords = geometry_from_smiles(smiles)
    mol = to_rdkit(numbers, coords, smiles=smiles)
    assert _isomeric(mol) == _isomeric(Chem.MolFromSmiles(smiles))


def test_strict_default_and_explicit_connectivity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Only explicit permissive mode reaches connectivity-only fallback."""
    monkeypatch.setattr(converter, "_from_xyz", Mock(side_effect=ValueError("No bond orders")))
    monkeypatch.setattr(converter, "_from_legacy", Mock(return_value=None))
    fallback_mol = Chem.MolFromSmiles("O")
    fallback = Mock(return_value=(None, fallback_mol))
    monkeypatch.setattr(converter, "xyz2ac_obabel", fallback)
    with pytest.raises(SteamrollConversionError, match="Bond-order inference failed"):
        to_rdkit([8], [[0, 0, 0]])
    fallback.assert_not_called()
    assert (
        to_rdkit([8], [[0, 0, 0]], fail_without_bond_order=False, remove_Hs=False) is fallback_mol
    )


@pytest.mark.parametrize("permissive", [False, True])
def test_rdkit_budget_is_terminal(monkeypatch: pytest.MonkeyPatch, permissive: bool) -> None:
    """Real RDKit budget exhaustion never restarts another backend."""
    monkeypatch.setattr(converter, "_BOND_ORDER_MAX_ITERATIONS", 1)
    legacy = Mock(side_effect=AssertionError("Budget exhaustion must not retry"))
    monkeypatch.setattr(converter, "_from_legacy", legacy)
    numbers, coords = geometry_from_smiles(_MACROCYCLE_SMILES)
    charge = 0
    with pytest.raises(SteamrollConversionError, match="iteration budget exhausted"):
        to_rdkit(numbers, coords, charge=charge, fail_without_bond_order=not permissive)
    legacy.assert_not_called()


def test_legacy_deadline_is_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Subprocess startup and inference share a deadline that propagates to callers."""
    monkeypatch.setattr(converter, "_from_xyz", Mock(side_effect=ValueError("No bond orders")))
    monkeypatch.setattr(converter, "_LEGACY_TIMEOUT_SECONDS", 0.000001)
    fallback = Mock(side_effect=AssertionError("Deadline must not trigger fallback"))
    monkeypatch.setattr(converter, "xyz2ac_obabel", fallback)
    with pytest.raises(SteamrollConversionError, match=r"Legacy.*budget exhausted"):
        to_rdkit([8], [[0, 0, 0]], fail_without_bond_order=False)
    fallback.assert_not_called()


def test_hypervalent_iodine_legacy_preserves_charge_and_coords() -> None:
    """Bounded legacy fallback retains the iodine regression's charge and coordinates."""
    numbers, coords, charge = read_xyz(DATA_DIR / "hypervalent_iodine.xyz")
    mol = to_rdkit(numbers, coords, charge=charge, remove_Hs=False)
    assert Chem.GetFormalCharge(mol) == charge
    np.testing.assert_allclose(mol.GetConformer().GetPositions(), coords, atol=1e-12)


@pytest.mark.parametrize("smiles", ["Oc1ccccn1", "NCC(=O)O", "NC=O"])
def test_guided_hydrogens_can_be_restored(smiles: str) -> None:
    """Removing explicit hydrogens retains counts needed to restore the molecular formula."""
    numbers, coords = geometry_from_smiles(smiles)
    template = Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(smiles)))
    mol = to_rdkit(numbers, coords, smiles=template, remove_Hs=True)
    restored = Chem.AddHs(mol)
    assert sorted(atom.GetAtomicNum() for atom in restored.GetAtoms()) == sorted(numbers)
    assert _isomeric(restored) == _isomeric(Chem.MolFromSmiles(smiles))


@pytest.mark.parametrize(
    "smiles",
    [
        "Cn1cnc2[nH]ncc2c1=N",
        "Cn1cnc2[nH]cnc2c1=N",
        "N=C(N)O",
        "Cn1ccc(=N)[nH]c1=O",
        "N=C(N)S",
        "CC(=N)NC(C)=O",
        "CC(=O)NC(=N)N",
        "N=C1CCC(=O)N1",
        "N=C1Cc2ccccc2N1",
    ],
)
def test_guided_imine_with_unspecified_stereo(smiles: str) -> None:
    """Coordinate-derived imine stereo does not change reference hydrogen topology."""
    numbers, coords = geometry_from_smiles(smiles)
    template = Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(smiles)))
    mol = to_rdkit(numbers, coords, smiles=template, remove_Hs=True)
    assert sorted(atom.GetAtomicNum() for atom in Chem.AddHs(mol).GetAtoms()) == sorted(numbers)
    assert converter._smiles_matches(mol, template)
