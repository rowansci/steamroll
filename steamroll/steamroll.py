"""steamroll package."""

import logging
import os
import tempfile
from collections import Counter
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D

from .utils import strip_to_connectivity
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

    Uses DetermineConnectivity + bond-type-agnostic substructure match to map XYZ
    atoms onto the SMILES template. The result is built from the template, so extra
    bonds from distorted geometry never appear in the output.

    Args:
        smiles: SMILES string encoding the molecular topology.
        atomic_numbers: atomic numbers for each atom.
        coordinates: Cartesian coordinates for each atom, in Å.

    Returns:
        RDKit molecule with SMILES topology and XYZ coordinates.

    Raises:
        ValueError: if SMILES is invalid, atom counts don't match, elements differ,
            or no valid atom mapping can be found.
    """
    template = Chem.MolFromSmiles(smiles)
    if template is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    template = Chem.AddHs(template)

    n = template.GetNumAtoms()
    if n != len(atomic_numbers):
        raise ValueError(f"Atom count mismatch: SMILES has {n}, XYZ has {len(atomic_numbers)}")

    xyz_pos = np.array(coordinates)

    if Counter(template.GetAtomWithIdx(i).GetAtomicNum() for i in range(n)) != Counter(
        atomic_numbers
    ):
        raise ValueError("Element mismatch between SMILES and XYZ")

    raw = Chem.RWMol()
    raw_conf = Chem.Conformer(n)
    for i, z in enumerate(atomic_numbers):
        raw.AddAtom(Chem.Atom(z))
        raw_conf.SetAtomPosition(i, Point3D(*xyz_pos[i].tolist()))
    raw.AddConformer(raw_conf, assignId=True)
    rdDetermineBonds.DetermineConnectivity(raw)

    match = raw.GetSubstructMatch(strip_to_connectivity(template))
    if not match or len(match) != n:
        raise ValueError("Could not find a valid atom mapping between SMILES and XYZ")

    # Build result with atoms in XYZ order; bonds/charges come from template.
    # match[t_idx] = raw_idx, so inv_match[raw_idx] = t_idx.
    inv_match = [0] * n
    for t_idx, r_idx in enumerate(match):
        inv_match[r_idx] = t_idx

    result = Chem.RWMol()
    result_conf = Chem.Conformer(n)
    for raw_idx, t_idx in enumerate(inv_match):
        new_atom = Chem.Atom(atomic_numbers[raw_idx])
        new_atom.SetFormalCharge(template.GetAtomWithIdx(t_idx).GetFormalCharge())
        result.AddAtom(new_atom)
        result_conf.SetAtomPosition(raw_idx, Point3D(*xyz_pos[raw_idx].tolist()))
    result.AddConformer(result_conf, assignId=True)

    for bond in template.GetBonds():
        result.AddBond(
            match[bond.GetBeginAtomIdx()],
            match[bond.GetEndAtomIdx()],
            bond.GetBondType(),
        )

    return result.GetMol()


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
        # isomericSmiles=False: only check connectivity, not stereo/isotopes
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
        return remove_hydrogens(rdkm) if remove_Hs else Chem.AddHs(rdkm, addCoords=True)

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
