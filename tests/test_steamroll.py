"""Tests for the steamroll package."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from steamroll.steamroll import ATOMIC_NUMBERS, SteamrollTopologyMismatchError, fragment, to_rdkit

_BROMOBENZENE_SMILES = "Brc1ccccc1"

# Bromobenzene with C-Br shrunk to 1.3 Å (normal ~1.9 Å); pulls Br into
# bonding range of the ortho carbons, causing DetermineConnectivity to mis-bond.
_BROMOBENZENE_DISTORTED_ATOMIC_NUMBERS = [35, 6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1]
_BROMOBENZENE_DISTORTED_COORDS = [
    [0.000, 2.700, 0.000],  # Br — 1.3 Å from C1 (distorted)
    [0.000, 1.400, 0.000],  # C1
    [1.212, 0.700, 0.000],  # C2
    [1.212, -0.700, 0.000],  # C3
    [0.000, -1.400, 0.000],  # C4
    [-1.212, -0.700, 0.000],  # C5
    [-1.212, 0.700, 0.000],  # C6
    [2.147, 1.240, 0.000],  # H on C2
    [2.147, -1.240, 0.000],  # H on C3
    [0.000, -2.480, 0.000],  # H on C4
    [-2.147, -1.240, 0.000],  # H on C5
    [-2.147, 1.240, 0.000],  # H on C6
]


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
    atomic_numbers = _BROMOBENZENE_DISTORTED_ATOMIC_NUMBERS
    coordinates = _BROMOBENZENE_DISTORTED_COORDS
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


def test_smiles_mismatch_raises() -> None:
    """Raises SteamrollTopologyMismatchError when no method can match the provided SMILES."""
    with pytest.raises(SteamrollTopologyMismatchError):
        to_rdkit([1, 8, 1], [[0, 0, 0], [0, 0, 1], [0, 1, 1]], smiles="CC")


@pytest.mark.parametrize(
    "smiles",
    [
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
