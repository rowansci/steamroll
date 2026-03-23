"""steamroll package."""

import logging
import os
import tempfile
from collections import Counter, defaultdict
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D
from scipy.optimize import linear_sum_assignment

from .xyz2mol.xyz2mol import xyz2mol
from .xyz2mol_tmc.xyz2mol_local import xyz2AC_obabel as xyz2ac_obabel
from .xyz2mol_tmc.xyz2mol_tmc import TRANSITION_METALS_NUM, get_tmc_mol

logger = logging.getLogger(__name__)

# Lanthanides and actinides are now included in TRANSITION_METALS_NUM, so
# has_tm will be True for them and they route through get_tmc_mol directly.
_SKIP_XYZ2MOL: frozenset[int] = frozenset()


class SteamrollConversionError(Exception):
    """Raised when a conversion error occurs."""


class SteamrollTopologyMismatchError(SteamrollConversionError):
    """Raised when conversion succeeds but the result doesn't match the provided SMILES."""


def remove_hydrogens(molecule: Chem.rdchem.Mol) -> Chem.rdchem.Mol:
    """Remove hydrogens from an RDKit molecule.

    Args:
        molecule: molecule

    Returns:
        RDKit molecule without hydrogens
    """
    rwmol = Chem.RWMol(molecule)

    # Iterate backwards to avoid messing indexing up. this is annoying
    for idx in range(rwmol.GetNumAtoms() - 1, -1, -1):
        atom = rwmol.GetAtomWithIdx(idx)

        # Delete hydrogen add an explicit H to its first neighbor
        if atom.GetAtomicNum() == 1:
            if neighbors := atom.GetNeighbors():
                neighbor = neighbors[0]
                rwmol.RemoveAtom(idx)
                neighbor.SetNumExplicitHs(neighbor.GetNumExplicitHs() + 1)
            else:
                logger.warning("Hydrogen atom has no neighbors, skipping")

    return rwmol.GetMol()


def fragment(molecule: Chem.rdchem.Mol) -> list[Chem.rdchem.Mol]:
    """Fragment an RDKit molecule.

    Args:
        molecule: molecule

    Returns:
        list of fragment molecules
    """
    return Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)  # type: ignore [return-value]


def _write_temp_xyz(atomic_numbers: list[int], coordinates: list[list[float]]) -> str:
    """Write atomic numbers and coordinates to a temporary xyz file.

    Args:
        atomic_numbers: atomic numbers for each atom
        coordinates: Cartesian coordinates for each atom, in Å

    Returns:
        path to the temporary file (caller is responsible for deletion)
    """
    pt = Chem.GetPeriodicTable()
    lines = [str(len(atomic_numbers)), ""]
    for num, (x, y, z) in zip(atomic_numbers, coordinates, strict=True):
        symbol = pt.GetElementSymbol(num)
        lines.append(f"{symbol}  {x}  {y}  {z}")
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False)
    f.write("\n".join(lines) + "\n")
    f.close()
    return f.name


def _from_smiles_and_coords(
    smiles: str,
    atomic_numbers: list[int],
    coordinates: list[list[float]],
) -> Chem.rdchem.Mol:
    """Build an RDKit mol using SMILES for topology and XYZ for 3D coordinates.

    Uses topology-driven BFS propagation with per-element optimal assignment to
    map template atoms to XYZ positions without relying on bond perception.
    Unique-element atoms serve as unambiguous anchors; the molecular graph then
    constrains candidates, and minimum-weight bipartite matching resolves
    ambiguity globally rather than greedily.

    Args:
        smiles: SMILES string encoding the molecular topology.
        atomic_numbers: atomic numbers for each atom.
        coordinates: Cartesian coordinates for each atom, in Å.

    Returns:
        RDKit molecule with SMILES topology and XYZ coordinates.

    Raises:
        ValueError: if SMILES is invalid, atom counts don't match, or elements differ.
    """
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    template = Chem.AddHs(template)

    n = template.GetNumAtoms()
    if n != len(atomic_numbers):
        raise ValueError(f"Atom count mismatch: SMILES has {n}, XYZ has {len(atomic_numbers)}")

    xyz_pos = np.array(coordinates)
    t_elems = [template.GetAtomWithIdx(i).GetAtomicNum() for i in range(n)]
    x_elems = list(atomic_numbers)

    if Counter(t_elems) != Counter(x_elems):
        raise ValueError("Element mismatch between SMILES and XYZ")

    t_by_elem: dict[int, list[int]] = defaultdict(list)
    x_by_elem: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        t_by_elem[t_elems[i]].append(i)
        x_by_elem[x_elems[i]].append(i)

    # Unique-element atoms are unambiguous anchors
    atom_map: dict[int, int] = {}
    for elem, t_indices_for_elem in t_by_elem.items():
        if len(t_indices_for_elem) == 1:
            atom_map[t_indices_for_elem[0]] = x_by_elem[elem][0]

    # BFS: expand frontier along bonds, assign each layer with min-weight bipartite matching
    changed = True
    while changed:
        changed = False

        frontier: dict[int, list[tuple[int, np.ndarray]]] = defaultdict(list)
        for t_idx in range(n):
            if t_idx in atom_map:
                continue
            elem = t_elems[t_idx]
            mapped_nbr_positions = [
                xyz_pos[atom_map[nbr.GetIdx()]]
                for nbr in template.GetAtomWithIdx(t_idx).GetNeighbors()
                if nbr.GetIdx() in atom_map
            ]
            if mapped_nbr_positions:
                frontier[elem].append((t_idx, np.mean(mapped_nbr_positions, axis=0)))

        for elem, candidates in frontier.items():
            unmapped_x = [x for x in x_by_elem[elem] if x not in atom_map.values()]
            if not unmapped_x:
                continue

            cost = np.array(
                [
                    [float(np.linalg.norm(xyz_pos[x_idx] - ref)) for x_idx in unmapped_x]
                    for _, ref in candidates
                ]
            )
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind, strict=True):
                atom_map[candidates[r][0]] = unmapped_x[c]
                changed = True

    # Fallback for atoms disconnected from all anchors (rare): embed and match
    unmapped_t = [i for i in range(n) if i not in atom_map]
    if unmapped_t:
        coord_map = {t_idx: Point3D(*xyz_pos[x_idx].tolist()) for t_idx, x_idx in atom_map.items()}
        AllChem.EmbedMolecule(template, randomSeed=42, coordMap=coord_map)  # type: ignore [attr-defined]
        t_pos = np.array([list(template.GetConformer().GetAtomPosition(i)) for i in range(n)])

        by_elem: dict[int, tuple[list[int], list[int]]] = {}
        for t_idx in unmapped_t:
            elem = t_elems[t_idx]
            if elem not in by_elem:
                by_elem[elem] = ([], [x for x in x_by_elem[elem] if x not in atom_map.values()])
            by_elem[elem][0].append(t_idx)

        for _elem, (t_indices, x_indices) in by_elem.items():
            cost = np.array(
                [
                    [float(np.linalg.norm(t_pos[t_idx] - xyz_pos[x_idx])) for x_idx in x_indices]
                    for t_idx in t_indices
                ]
            )
            row_ind, col_ind = linear_sum_assignment(cost)
            for r, c in zip(row_ind, col_ind, strict=True):
                atom_map[t_indices[r]] = x_indices[c]

    conf = Chem.Conformer(n)
    for t_idx in range(n):
        conf.SetAtomPosition(t_idx, Point3D(*xyz_pos[atom_map[t_idx]].tolist()))
    template.RemoveAllConformers()
    template.AddConformer(conf, assignId=True)
    return template


def _smiles_matches(mol: Chem.rdchem.Mol, smiles: str) -> bool:
    """Check whether an RDKit molecule's topology matches a SMILES string.

    Args:
        mol: RDKit molecule to validate.
        smiles: reference SMILES string.

    Returns:
        True if canonical SMILES match after stripping hydrogens.
    """
    try:
        ref = Chem.MolFromSmiles(smiles)
        if ref is None:
            return False
        Chem.SanitizeMol(mol)
        # isomericSmiles=False strips stereo/isotopes — we only care about connectivity
        got = Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=False)
        return got == Chem.MolToSmiles(ref, isomericSmiles=False)
    except Exception:
        return False


def to_rdkit(
    atomic_numbers: Iterable[int],
    coordinates: ArrayLike,
    charge: int = 0,
    remove_Hs: bool = True,
    fail_without_bond_order: bool = False,
    smiles: str | None = None,
) -> Chem.rdchem.Mol:
    """Convert a given molecular geometry to an RDKit molecule.

    When ``smiles`` is provided, topology-driven coordinate assignment is attempted
    first and all subsequent methods are validated against the SMILES; any method
    that produces a non-matching topology is skipped, and ``SteamrollTopologyMismatchError``
    is raised if no method matches.

    Args:
        atomic_numbers: atomic numbers
        coordinates: coordinates, in Å
        charge: charge
        remove_Hs: whether or not to strip hydrogens from the output molecule
        fail_without_bond_order: if bond order cannot be detected, raise SteamrollConversionError
        smiles: optional SMILES string; used as the primary conversion method
            and as a topology validator for all fallback methods

    Raises:
        ValueError: if input dimensions aren't correct
        SteamrollConversionError: if conversion fails
        SteamrollTopologyMismatchError: if smiles is provided but no method produces a
            matching topology

    Returns:
        RDKit molecule
    """
    atomic_numbers = list(atomic_numbers)
    coordinates = np.asarray(coordinates)

    if coordinates.ndim != 2:
        raise ValueError("`coordinates` needs to be a two-dimensional")
    if coordinates.shape[1] != 3:
        raise ValueError("Coordinates needs to have second dimension with length 3")
    if (n_atoms := len(atomic_numbers)) != (n_coords := len(coordinates)):
        raise ValueError(
            f"Length of atomic numbers ({n_atoms}) doesn't match coordinates ({n_coords})"
        )

    coords = coordinates.tolist()
    has_tm = any(n in TRANSITION_METALS_NUM for n in atomic_numbers)
    has_exotic = any(n in _SKIP_XYZ2MOL for n in atomic_numbers)

    # SMILES-based method: topology from SMILES, positions from XYZ
    if smiles is not None:
        try:
            rdkm = _from_smiles_and_coords(smiles, atomic_numbers, coords)
            if _smiles_matches(rdkm, smiles):
                return remove_hydrogens(rdkm) if remove_Hs else rdkm
            logger.debug("SMILES-based conversion produced wrong topology, falling back")
        except Exception as e:
            logger.debug(f"SMILES-based conversion failed, falling back: {e}")

    rdkm: Chem.rdchem.Mol | None = None

    if has_tm:
        # Use the specialized TMC converter; Hs come back implicit → make explicit.
        xyz_file = _write_temp_xyz(atomic_numbers, coords)
        try:
            rdkm = get_tmc_mol(xyz_file, charge)
        except Exception as e:
            raise SteamrollConversionError("xyz2mol_tm conversion failed") from e
        finally:
            os.unlink(xyz_file)
        if rdkm is None:
            raise SteamrollConversionError("xyz2mol_tm returned no molecule")
        return Chem.AddHs(rdkm)

    def _topology_ok(mol: Chem.rdchem.Mol) -> bool:
        return smiles is None or _smiles_matches(mol, smiles)

    if not has_exotic:
        # xyz2mol (standard)
        try:
            candidate = xyz2mol(atomic_numbers, coords, charge=charge)[0]
            if _topology_ok(candidate):
                rdkm = candidate
            else:
                logger.debug("xyz2mol produced wrong topology, trying Hückel")
        except Exception:
            logger.debug("xyz2mol failed, trying Hückel")

        # xyz2mol (Hückel) — if standard failed or gave wrong topology
        if rdkm is None:
            try:
                candidate = xyz2mol(atomic_numbers, coords, charge=charge, use_huckel=True)[0]
                if _topology_ok(candidate):
                    rdkm = candidate
                else:
                    logger.debug("xyz2mol Hückel produced wrong topology, trying obabel")
            except Exception:
                logger.debug("xyz2mol Hückel failed, trying obabel")

        if rdkm is None and fail_without_bond_order:
            raise SteamrollConversionError(
                f"xyz2mol failed for {len(atomic_numbers)}-atom molecule (charge={charge}); "
                "provide a SMILES string or fix the geometry"
            )

    if rdkm is None:
        # Geometry-only fallback via obabel — no bond orders, last resort.
        try:
            _, rdkm = xyz2ac_obabel(atomic_numbers, coords)
        except Exception as e:
            raise SteamrollConversionError(
                f"all conversion methods failed for {len(atomic_numbers)}-atom molecule "
                f"(charge={charge}); provide a SMILES string or fix the geometry"
            ) from e
        if not _topology_ok(rdkm):
            try:
                got = Chem.MolToSmiles(Chem.RemoveHs(rdkm), isomericSmiles=False)
            except Exception:
                got = "<could not determine>"
            expected = Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=False)
            raise SteamrollTopologyMismatchError(
                f"no conversion method matched the provided SMILES for "
                f"{len(atomic_numbers)}-atom molecule (charge={charge})\n"
                f"  expected: {expected}\n"
                f"  got:      {got}"
            )

    return remove_hydrogens(rdkm) if remove_Hs else rdkm


ATOMIC_NUMBERS = {
    "X": 0,
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
    "Md": 101,
    "No": 102,
    "Lr": 103,
    "Rf": 104,
    "Db": 105,
    "Sg": 106,
    "Bh": 107,
    "Hs": 108,
    "Mt": 109,
    "Ds": 110,
    "Rg": 111,
    "Cp": 112,
    "Uut": 113,
    "Uuq": 114,
    "Uup": 115,
    "Uuh": 116,
    "Uus": 117,
    "Uuo": 118,
}
