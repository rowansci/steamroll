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
from steamroll.utils import strip_to_connectivity

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


def geometry_from_smiles(smiles: str, seed: int = 42) -> tuple[list[int], list[list[float]]]:
    """Generate deterministic 3D coordinates for an explicit-hydrogen molecule."""
    ref = Chem.MolFromSmiles(smiles)
    assert ref is not None
    mol_h = Chem.AddHs(ref)
    assert AllChem.EmbedMolecule(mol_h, randomSeed=seed) == 0  # type: ignore[attr-defined]
    AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)  # type: ignore[attr-defined]

    conf = mol_h.GetConformer()
    atomic_numbers = [a.GetAtomicNum() for a in mol_h.GetAtoms()]
    coordinates = [
        [conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y, conf.GetAtomPosition(i).z]
        for i in range(mol_h.GetNumAtoms())
    ]
    return atomic_numbers, coordinates


def canonical_smiles(mol: Chem.rdchem.Mol | str) -> str:
    """Canonical non-isomeric heavy-atom SMILES."""
    rdkm = Chem.MolFromSmiles(mol) if isinstance(mol, str) else Chem.RemoveHs(mol)
    assert rdkm is not None
    return Chem.MolToSmiles(rdkm, isomericSmiles=False)


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


@pytest.mark.parametrize(
    ("name", "smiles", "charge", "expected_smiles"),
    [
        ("DMSO", "CS(C)=O", 0, "CS(C)=O"),
        ("3-methylthiazolium", "C[n+]1ccsc1", 1, "C[n+]1ccsc1"),
        ("2-thiophene carboxylic acid", "O=C(O)c1cccs1", 0, "O=C(O)c1cccs1"),
    ],
)
def test_xyz_only_heteroatom_charge_regressions(
    name: str,
    smiles: str,
    charge: int,
    expected_smiles: str,
) -> None:
    """XYZ-only conversion prefers chemically reasonable heteroatom charges."""
    atomic_numbers, coordinates = geometry_from_smiles(smiles, seed=17)
    rdkm = to_rdkit(atomic_numbers, coordinates, charge=charge, remove_Hs=False)

    assert canonical_smiles(rdkm) == canonical_smiles(expected_smiles), name

    charges = {
        (atom.GetSymbol(), atom.GetIdx()): atom.GetFormalCharge()
        for atom in rdkm.GetAtoms()
        if atom.GetFormalCharge()
    }
    if name == "DMSO":
        assert charges == {}
    elif name == "3-methylthiazolium":
        assert any(
            atom.GetAtomicNum() == 7 and atom.GetFormalCharge() == 1 for atom in rdkm.GetAtoms()
        )
        assert all(
            atom.GetFormalCharge() == 0 for atom in rdkm.GetAtoms() if atom.GetAtomicNum() == 16
        )


def test_xyz_only_hypervalent_iodine_regression() -> None:
    """Charge-penalized fallback must not remove existing hypervalent iodine support."""
    atomic_numbers, coordinates, charge = read_xyz(DATA_DIR / "hypervalent_iodine.xyz")
    rdkm = to_rdkit(atomic_numbers, coordinates, charge=charge, remove_Hs=False)

    assert rdkm.GetNumAtoms() == len(atomic_numbers)
    assert sum(atom.GetAtomicNum() == 53 for atom in rdkm.GetAtoms()) == 1


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


def test_strip_to_connectivity_clears_radicals_and_aromaticity() -> None:
    """Connectivity-only matching ignores radical and aromatic annotations."""
    radical = strip_to_connectivity(Chem.MolFromSmiles("[CH3]"))
    assert radical.GetAtomWithIdx(0).GetNumRadicalElectrons() == 0

    aromatic = strip_to_connectivity(Chem.MolFromSmiles("c1ccccc1"))
    assert all(not atom.GetIsAromatic() for atom in aromatic.GetAtoms())
    assert all(not bond.GetIsAromatic() for bond in aromatic.GetBonds())
    assert all(bond.GetBondType() == Chem.BondType.SINGLE for bond in aromatic.GetBonds())


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


def test_from_smiles_and_coords_heavy_atom_only() -> None:
    """_from_smiles_and_coords works when coordinates contain only heavy atoms (no H).

    This is the typical case when loading from a PDB file: the XYZ has no explicit
    H atoms, but the SMILES encodes bond orders for the heavy-atom skeleton.
    """
    # Ethanol heavy atoms only: C, C, O
    smiles = "CCO"
    atomic_numbers = [6, 6, 8]
    # Rough heavy-atom geometry for ethanol
    coordinates = [[0.0, 0.0, 0.0], [1.54, 0.0, 0.0], [2.4, 1.1, 0.0]]
    mol = _from_smiles_and_coords(smiles, atomic_numbers, coordinates)
    assert mol.GetNumAtoms() == 3
    assert mol.GetNumBonds() == 2
    elements = sorted(a.GetAtomicNum() for a in mol.GetAtoms())
    assert elements == [6, 6, 8]


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

_BZU_PDB_BLOCK = """\
HETATM    2  C9  BZU A 555      -3.394   5.205  13.019  1.00  0.00           C
HETATM    3  C10 BZU A 555      -3.055   3.867  13.710  1.00  0.00           C
HETATM    4  C11 BZU A 555      -3.349   7.408  13.935  1.00  0.00           C
HETATM    5  C12 BZU A 555      -2.076   7.426  14.764  1.00  0.00           C
HETATM    6  C14 BZU A 555      -1.915   9.769  15.059  1.00  0.00           C
HETATM    7  C15 BZU A 555      -0.878  10.903  15.178  1.00  0.00           C
HETATM    8  O1A BZU A 555      -5.425   0.994  18.287  1.00  0.00           O
HETATM    9  O2A BZU A 555      -7.398   0.817  16.995  1.00  0.00           O
HETATM   10  N21 BZU A 555      -5.486  -0.545  16.461  1.00  0.00           N
HETATM   11  S1  BZU A 555      -5.966   0.750  16.997  1.00  0.00           S
HETATM   12  C4  BZU A 555      -4.321   2.152  15.292  1.00  0.00           C
HETATM   13  C5  BZU A 555      -4.165   3.345  14.642  1.00  0.00           C
HETATM   14  C6  BZU A 555      -5.216   4.200  14.931  1.00  0.00           C
HETATM   15  S2  BZU A 555      -6.379   3.501  15.979  1.00  0.00           S
HETATM   16  S7  BZU A 555      -5.295   5.827  14.368  1.00  0.00           S
HETATM   17  O3B BZU A 555      -6.160   5.873  13.244  1.00  0.00           O
HETATM   18  O4B BZU A 555      -5.858   6.680  15.373  1.00  0.00           O
HETATM   19  N8  BZU A 555      -3.848   6.065  14.090  1.00  0.00           N
HETATM   20  N16 BZU A 555      -2.776   2.895  12.649  1.00  0.00           N
HETATM   21  O13 BZU A 555      -1.307   8.596  14.533  1.00  0.00           O
HETATM   22  C17 BZU A 555      -1.564   3.011  11.841  1.00  0.00           C
HETATM   23  C18 BZU A 555      -1.463   1.927  10.754  1.00  0.00           C
"""


def test_pdb_heavy_atom_only_gets_bonds() -> None:
    """Regression test: heavy-atom-only input (e.g. from a PDB file) must yield
    a connected molecule with proper bond orders, not isolated atoms.

    Before the fix, xyz2mol would silently fail on inputs with no explicit
    hydrogens and the obabel fallback would return atoms with no bonds, giving
    a SMILES like ``C.C.C.C...``.  After the fix, rdDetermineBonds.DetermineBonds
    handles the heavy-atom-only case and assigns connectivity + bond orders.
    """
    pdb_mol = Chem.MolFromPDBBlock(_BZU_PDB_BLOCK, removeHs=False)
    assert pdb_mol is not None, "RDKit could not parse the PDB block"

    crds = pdb_mol.GetConformer(0).GetPositions().tolist()
    atomic_nums = [atom.GetAtomicNum() for atom in pdb_mol.GetAtoms()]

    # Confirm the input really is heavy-atom-only (no hydrogens)
    assert 1 not in atomic_nums, "Expected no explicit H atoms in this PDB input"

    rd_mol = to_rdkit(atomic_nums, crds, charge=0)

    # All 22 heavy atoms must be present
    assert rd_mol.GetNumAtoms() == 22

    # Every atom must have at least one bond — the core regression check
    isolated = [
        atom.GetIdx()
        for atom in rd_mol.GetAtoms()
        if atom.GetDegree() == 0
    ]
    assert isolated == [], f"Atoms with no bonds (isolated): {isolated}"

    # The SMILES must not be the all-disconnected pattern produced by the bug
    smiles = Chem.MolToSmiles(rd_mol)
    assert "." not in smiles or smiles.count(".") < rd_mol.GetNumAtoms() - 1, (
        f"Molecule looks disconnected: {smiles}"
    )

    # Canonical heavy-atom SMILES must match the topology reported by RDKit's own
    # PDB parser.  We compare non-isomeric SMILES after stripping any remaining
    # explicit Hs so that bookkeeping differences (e.g. "[SH]" vs "S" with an
    # implicit H) do not cause spurious mismatches.
    ref_smiles = Chem.MolToSmiles(
        Chem.RemoveHs(Chem.RWMol(pdb_mol).GetMol()), isomericSmiles=False
    )
    got_smiles = Chem.MolToSmiles(rd_mol, isomericSmiles=False)
    assert got_smiles == ref_smiles, (
        f"SMILES mismatch:\n  expected (PDB parser): {ref_smiles}\n  got (steamroll):       {got_smiles}"
    )
