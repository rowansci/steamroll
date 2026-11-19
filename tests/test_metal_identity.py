"""Metal conversion retains atom identity through chemistry corrections."""

from unittest.mock import Mock

import numpy as np
import pytest
from rdkit import Chem

from steamroll import to_rdkit
from steamroll.xyz2mol_tmc import xyz2mol_tmc
from tests.test_steamroll import DATA_DIR, read_xyz

_METAL_FIXTURES = [
    "zinc_chloride",
    "zinc_ammonia",
    "ferrocene",
    "UCl6",
    "Ce_18-crown-6",
    "PdCl2MeCN2",
    "ethyl_methyl_cobaltacene",
    "fe_pyridone_complex",
]


@pytest.mark.parametrize("name", _METAL_FIXTURES)
@pytest.mark.parametrize("remove_hs", [False, True])
@pytest.mark.parametrize("shuffle", [False, True])
def test_metal_atom_identity(name: str, remove_hs: bool, shuffle: bool) -> None:
    """Every retained atom has its input element, position, and relative order."""
    numbers, coords, charge = read_xyz(DATA_DIR / f"{name}.xyz")
    if shuffle:
        order = np.random.default_rng(19).permutation(len(numbers))
        numbers = [numbers[i] for i in order]
        coords = [coords[i] for i in order]
    mol = to_rdkit(numbers, coords, charge=charge, remove_Hs=remove_hs)
    kept = [i for i, number in enumerate(numbers) if not remove_hs or number != 1]
    assert [atom.GetAtomicNum() for atom in mol.GetAtoms()] == [numbers[i] for i in kept]
    np.testing.assert_array_equal(mol.GetConformer().GetPositions(), [coords[i] for i in kept])
    assert Chem.GetFormalCharge(mol) == charge
    Chem.SanitizeMol(mol)
    assert sorted(atom.GetAtomicNum() for atom in Chem.AddHs(mol).GetAtoms()) == sorted(numbers)


@pytest.mark.parametrize(
    ("name", "ligand_smiles", "metal_charge", "coordination"),
    [
        ("zinc_ammonia", "N", 2, 1),
        ("zinc_chloride", "[Cl-].[Cl-]", 2, 2),
        ("ferrocene", "[cH-]1cccc1.[cH-]1cccc1", 2, 10),
        ("PdCl2MeCN2", "CC#N.CC#N.[Cl-].[Cl-]", 2, 4),
    ],
)
def test_metal_ligand_chemistry(
    name: str, ligand_smiles: str, metal_charge: int, coordination: int
) -> None:
    """Metal conversion preserves ligand bond orders, charges, and coordination."""
    numbers, coords, charge = read_xyz(DATA_DIR / f"{name}.xyz")
    mol = to_rdkit(numbers, coords, charge=charge)
    (metal,) = (
        atom for atom in mol.GetAtoms() if atom.GetAtomicNum() in xyz2mol_tmc.TRANSITION_METALS_NUM
    )
    assert metal.GetFormalCharge() == metal_charge
    assert metal.GetDegree() == coordination
    assert all(bond.GetBondType() == Chem.BondType.DATIVE for bond in metal.GetBonds())
    ligands = Chem.RWMol(mol)
    ligands.RemoveAtom(metal.GetIdx())
    Chem.SanitizeMol(ligands)
    assert Chem.MolToSmiles(ligands) == Chem.MolToSmiles(Chem.MolFromSmiles(ligand_smiles))


@pytest.mark.parametrize("atomic_number", [9, 17, 35, 53])
def test_halide_charge_skips_native_huckel(
    monkeypatch: pytest.MonkeyPatch, atomic_number: int
) -> None:
    """Monatomic halides are assigned -1 without calling the native solver."""
    eht = Mock(side_effect=AssertionError("Monatomic halides do not need EHT"))
    monkeypatch.setattr(xyz2mol_tmc.rdEHTTools, "RunMol", eht)
    ligand = Chem.RWMol()
    ligand.AddAtom(Chem.Atom(atomic_number))
    assert xyz2mol_tmc.get_proposed_ligand_charge(ligand) == -1
    eht.assert_not_called()
