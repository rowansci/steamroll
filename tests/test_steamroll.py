"""Tests for the steamroll package."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from steamroll import (
    SteamrollTopologyMismatchError,
    to_rdkit,
)
from steamroll.steamroll import (
    ATOMIC_NUMBERS,
    _from_smiles_and_coords,
    fragment,
)

_BROMOBENZENE_SMILES = "Brc1ccccc1"
_NAPHTHALENE_SMILES = "c1ccc2ccccc2c1"


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
            atomic_numbers.append(int(atom) if atom.isdigit() else ATOMIC_NUMBERS[atom])
            coordinates.append([float(x), float(y), float(z)])
    return atomic_numbers, coordinates, charge


def test_steamroll() -> None:
    """Basic test to make sure the package is working."""
    rdkm = to_rdkit([1, 8, 1], [[0, 0, 0], [0, 0, 1], [0, 1, 1]])
    assert rdkm.GetNumAtoms() == 1


def test_no_remove_hydrogens() -> None:
    """Hydrogens are retained when remove_Hs=False."""
    rdkm = to_rdkit([1, 8, 1], [[0, 0, 0], [0, 0, 1], [0, 1, 1]], remove_Hs=False)
    assert rdkm.GetNumAtoms() == 3


def test_fragment() -> None:
    """Multiple molecules are correctly fragmented."""
    rdkm = to_rdkit(
        [1, 8, 1, 1, 8, 1],
        [[0, 0, 0], [0, 0, 1], [0, 1, 1], [50, 0, 0], [50, 0, 1], [50, 1, 1]],
    )
    rdkm1, rdkm2 = fragment(rdkm)
    assert rdkm1.GetNumAtoms() == 1
    assert rdkm2.GetNumAtoms() == 1


@pytest.mark.parametrize("file", DATA_DIR.glob("*.xyz"))
def test_all_data(file: str) -> None:
    """All data files can be processed."""
    atomic_numbers, coordinates, charge = read_xyz(file)
    rdkm = to_rdkit(atomic_numbers, coordinates, charge=charge, remove_Hs=False)
    assert rdkm.GetNumAtoms() == len(atomic_numbers)


def test_smiles_distorted_halogen() -> None:
    """SMILES-based conversion fixes Br bonding when geometry is distorted.

    Without SMILES, geometry-only methods return wrong topology (Br disconnected).
    With SMILES, the correct topology is recovered: Br has exactly 1 bond and
    all heavy-atom coordinates are preserved.
    """
    atomic_numbers, coordinates, _ = read_xyz(DATA_DIR / "bromobenzene_distorted.xyz")
    ref_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(_BROMOBENZENE_SMILES), isomericSmiles=False)

    # Without SMILES: wrong topology
    got_no_smiles = Chem.MolToSmiles(
        Chem.RemoveHs(to_rdkit(atomic_numbers, coordinates, remove_Hs=False)),
        isomericSmiles=False,
    )
    assert got_no_smiles != ref_smiles

    # With SMILES: correct topology, Br valence, and coordinate fidelity
    rdkm = to_rdkit(atomic_numbers, coordinates, smiles=_BROMOBENZENE_SMILES, remove_Hs=False)
    assert Chem.MolToSmiles(Chem.RemoveHs(rdkm), isomericSmiles=False) == ref_smiles
    br = next(a for a in rdkm.GetAtoms() if a.GetAtomicNum() == 35)
    assert br.GetDegree() == 1

    input_heavy = np.array(sorted([coordinates[i] for i, n in enumerate(atomic_numbers) if n > 1]))
    output_heavy = np.array(
        sorted(
            [
                (p.x, p.y, p.z)
                for a in rdkm.GetAtoms()
                if a.GetAtomicNum() > 1
                for p in [rdkm.GetConformer().GetAtomPosition(a.GetIdx())]
            ]
        )
    )
    np.testing.assert_allclose(input_heavy, output_heavy, atol=1e-3)


def test_smiles_fused_ring() -> None:
    """SMILES-based conversion correctly maps atoms in fused ring systems."""
    atomic_numbers, coordinates, _ = read_xyz(DATA_DIR / "naphthalene.xyz")
    ref_smiles = Chem.MolToSmiles(Chem.MolFromSmiles(_NAPHTHALENE_SMILES), isomericSmiles=False)
    rdkm = to_rdkit(atomic_numbers, coordinates, smiles=_NAPHTHALENE_SMILES, remove_Hs=False)
    assert Chem.MolToSmiles(Chem.RemoveHs(rdkm), isomericSmiles=False) == ref_smiles

    # Verify bond lengths match naphthalene geometry; wrong atom assignments
    # produce C-C bonds at ~2.4 Å and C-H bonds at ~7 Å.
    conf = rdkm.GetConformer()
    for bond in rdkm.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        d = conf.GetAtomPosition(i).Distance(conf.GetAtomPosition(j))
        is_ch = {rdkm.GetAtomWithIdx(i).GetAtomicNum(), rdkm.GetAtomWithIdx(j).GetAtomicNum()} == {
            6,
            1,
        }
        assert d < (1.3 if is_ch else 1.6), f"Bond {i}-{j}: {d:.2f} Å"


def test_smiles_mismatch_raises() -> None:
    """Raises SteamrollTopologyMismatchError when no method can match the provided SMILES."""
    with pytest.raises(SteamrollTopologyMismatchError):
        to_rdkit([1, 8, 1], [[0, 0, 0], [0, 0, 1], [0, 1, 1]], smiles="CC")


@pytest.mark.parametrize(
    "smiles",
    [
        "C=[N+]=[N-]",  # diazomethane (formal charges)
        "C[N+](=O)[O-]",  # nitromethane (formal charges)
        "[NH3+]CC(=O)[O-]",  # glycine zwitterion
        "C12C3C4C1C5C4C3C25",  # cubane
        "C1C2CC3CC1CC(C2)C3",  # adamantane
        "c1cc2ccc3ccc4ccc5ccc1c1c2c3c4c51",  # corannulene
        "CNC1(C2)CC2(C)C1",  # BCP derivative
    ],
)
def test_smiles_high_symmetry(smiles: str) -> None:
    """SMILES-guided assignment works on highly symmetric molecules.

    Generates RDKit 3D coords, re-converts with the SMILES, and verifies the
    canonical SMILES is preserved.
    """
    ref = Chem.MolFromSmiles(smiles)
    mol_h = Chem.AddHs(ref)
    AllChem.EmbedMolecule(mol_h, randomSeed=42)  # type: ignore [attr-defined]

    atomic_numbers = [a.GetAtomicNum() for a in mol_h.GetAtoms()]
    conf = mol_h.GetConformer()
    coordinates = [
        [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
        for i in range(mol_h.GetNumAtoms())
    ]

    rdkm = to_rdkit(atomic_numbers, coordinates, smiles=smiles, remove_Hs=False)
    assert Chem.MolToSmiles(Chem.RemoveHs(rdkm), isomericSmiles=False) == Chem.MolToSmiles(
        ref, isomericSmiles=False
    )


def test_from_smiles_and_coords_xyz_order() -> None:
    """_from_smiles_and_coords returns atoms in XYZ input order, not SMILES order.

    Water: SMILES "O" expands to O, H, H (oxygen first).  The XYZ supplies atoms
    in H, O, H order.  After the fix atom 0 must be H at (0,0,0), atom 1 must be
    O at (0,0,1), and atom 2 must be H at (0,1,1).
    """
    # XYZ order: H, O, H  (deliberately different from SMILES O-H-H order)
    atomic_numbers = [1, 8, 1]
    coordinates = [[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
    mol = _from_smiles_and_coords("O", atomic_numbers, coordinates)
    conf = mol.GetConformer()
    for i, (z, coord) in enumerate(zip(atomic_numbers, coordinates, strict=True)):
        assert mol.GetAtomWithIdx(i).GetAtomicNum() == z, f"atom {i}: wrong element"
        pos = conf.GetAtomPosition(i)
        np.testing.assert_allclose(
            [pos.x, pos.y, pos.z], coord, atol=1e-6, err_msg=f"atom {i}: wrong position"
        )


def test_from_smiles_and_coords_charges_preserved() -> None:
    """Formal charges from SMILES are transferred to the correct XYZ-ordered atoms.

    Glycine zwitterion [NH3+]CC(=O)[O-]: the positive charge belongs to N and the
    negative charge to one O.  We give coordinates in an order that differs from the
    SMILES template and verify the charges end up on the right elements.
    """
    # Build a reference mol with 3D coords in RDKit's own order
    smiles = "[NH3+]CC(=O)[O-]"
    ref = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(ref, randomSeed=0)  # type: ignore[attr-defined]
    atomic_numbers = [a.GetAtomicNum() for a in ref.GetAtoms()]
    conf_ref = ref.GetConformer()
    coordinates = [
        [
            conf_ref.GetAtomPosition(i).x,
            conf_ref.GetAtomPosition(i).y,
            conf_ref.GetAtomPosition(i).z,
        ]
        for i in range(ref.GetNumAtoms())
    ]

    # Shuffle input order so it differs from the SMILES template order
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(atomic_numbers)).tolist()
    shuffled_nums = [atomic_numbers[p] for p in perm]
    shuffled_coords = [coordinates[p] for p in perm]

    mol = _from_smiles_and_coords(smiles, shuffled_nums, shuffled_coords)
    result_conf = mol.GetConformer()

    # Every atom's position must match its shuffled input coordinate
    for i in range(mol.GetNumAtoms()):
        pos = result_conf.GetAtomPosition(i)
        np.testing.assert_allclose(
            [pos.x, pos.y, pos.z],
            shuffled_coords[i],
            atol=1e-6,
            err_msg=f"atom {i}: wrong position",
        )

    # Formal charges must land on the right elements
    charges_by_element: dict[int, list[int]] = {}
    for atom in mol.GetAtoms():
        charges_by_element.setdefault(atom.GetAtomicNum(), []).append(atom.GetFormalCharge())
    assert 1 in charges_by_element[7], "N should have +1 formal charge"
    assert -1 in charges_by_element[8], "one O should have -1 formal charge"


def test_tmc_conformer_preserved() -> None:
    """to_rdkit preserves 3D coordinates for transition metal complexes.

    Previously get_tmc_mol discarded coordinates by roundtripping through SMILES,
    returning a mol with no conformer.
    """
    atomic_numbers, coordinates, charge = read_xyz(DATA_DIR / "fe_pyridone_complex.xyz")
    rdkm = to_rdkit(atomic_numbers, coordinates, charge=charge, remove_Hs=True)
    assert rdkm.GetNumConformers() == 1
    conf = rdkm.GetConformer()
    positions = [conf.GetAtomPosition(i) for i in range(rdkm.GetNumAtoms())]
    assert not all(p.x == 0.0 and p.y == 0.0 and p.z == 0.0 for p in positions)
