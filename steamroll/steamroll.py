"""steamroll package."""

import logging
import os
import tempfile
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike
from rdkit import Chem

from .xyz2mol.xyz2mol import xyz2mol
from .xyz2mol_tmc.xyz2mol_local import xyz2AC_obabel as xyz2ac_obabel
from .xyz2mol_tmc.xyz2mol_tmc import TRANSITION_METALS_NUM, get_tmc_mol

logger = logging.getLogger(__name__)

# Lanthanides Ce-Yb (58-70) and actinides Ac-Lr (89-103) that xyz2mol cannot
# handle. Molecules containing these elements bypass xyz2mol and go directly to
# the geometry-only xyz2ac_obabel fallback.
_SKIP_XYZ2MOL: frozenset[int] = frozenset(range(58, 71)) | frozenset(range(89, 104))


class SteamrollConversionError(Exception):
    """Raised when a conversion error occurs."""


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
            if neighbors := atom.GetNeighbors():  # type: ignore [call-arg, unused-ignore]
                neighbor = neighbors[0]
                rwmol.RemoveAtom(idx)
                neighbor.SetNumExplicitHs(neighbor.GetNumExplicitHs() + 1)
            else:
                logger.warning("Hydrogen atom has no neighbors, skipping")

    return rwmol.GetMol()  # type: ignore [call-arg, no-any-return, unused-ignore]


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


def to_rdkit(
    atomic_numbers: Iterable[int],
    coordinates: ArrayLike,
    charge: int = 0,
    remove_Hs: bool = True,
    fail_without_bond_order: bool = False,
) -> Chem.rdchem.Mol:
    """Convert a given molecular geometry to an RDKit molecule.

    Args:
        atomic_numbers: atomic numbers
        coordinates: coordinates, in Å
        charge: charge
        remove_Hs: whether or not to strip hydrogens from the output molecule
        fail_without_bond_order: if bond order cannot be detected, raise SteamrollConversionError

    Raises:
        ValueError: if input dimensions aren't correct
        SteamrollConversionError: if conversion fails

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

    rdkm: Chem.rdchem.Mol | None = None

    if not has_tm and not has_exotic:
        try:
            try:
                rdkm = xyz2mol(atomic_numbers, coords, charge=charge)[0]
            except (Exception, ValueError, IndexError):
                rdkm = xyz2mol(atomic_numbers, coords, charge=charge, use_huckel=True)[0]
        except Exception as e:
            if fail_without_bond_order:
                raise SteamrollConversionError from e

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
        return Chem.AddHs(rdkm)  # type: ignore [return-value]

    if rdkm is None:
        # xyz2mol failed for a non-TM molecule (e.g. wrong charge, unsupported
        # element). Fall back to a geometry-only mol via obabel connectivity.
        try:
            _, rdkm = xyz2ac_obabel(atomic_numbers, coords)
        except Exception as e:
            raise SteamrollConversionError("xyz2mol conversion failed") from e
        return rdkm  # type: ignore [return-value]

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
