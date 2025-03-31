"""steamroll package."""

import logging
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike
from rdkit import Chem

from .xyz2mol import xyz2mol

logger = logging.getLogger(__name__)


class SteamrollConversionError(Exception):
    """Raised when a conversion error occurs."""


def remove_hydrogens(molecule: Chem.rdchem.Mol) -> Chem.rdchem.Mol:
    """
    Remove hydrogens from an RDKit molecule.

    :param molecule: molecule
    :returns: RDKit molecule without hydrogens
    """
    rwmol = Chem.RWMol(molecule)

    # Iterate backwards to avoid messing indexing up. this is annoying
    for idx in range(rwmol.GetNumAtoms() - 1, -1, -1):
        atom = rwmol.GetAtomWithIdx(idx)

        # Delete hydrogen add an explicit H to its first neighbor
        if atom.GetAtomicNum() == 1:
            if neighbors := atom.GetNeighbors():  # type: ignore [call-arg]
                neighbor = neighbors[0]
                rwmol.RemoveAtom(idx)
                neighbor.SetNumExplicitHs(neighbor.GetNumExplicitHs() + 1)
            else:
                logger.warning("Hydrogen atom has no neighbors, skipping")

    return rwmol.GetMol()  # type: ignore [call-arg, no-any-return]


def fragment(molecule: Chem.rdchem.Mol) -> list[Chem.rdchem.Mol]:
    """
    Fragment an RDKit molecule.

    :param molecule: molecule
    :returns: list of fragment molecules
    """
    return Chem.GetMolFrags(molecule, asMols=True, sanitizeFrags=True)  # type: ignore [return-value]


def to_rdkit(
    atomic_numbers: Iterable[int],
    coordinates: ArrayLike,
    charge: int = 0,
    remove_Hs: bool = True,
) -> Chem.rdchem.Mol:
    """
    Convert a given molecular geometry to an RDKit molecule.

    :param atomic_numbers: atomic numbers
    :param coordinates: coordinates, in Å
    :param charge: charge
    :param remove_Hs: whether or not to strip hydrogens from the output molecule
    :raises ValueError: if input dimensions aren't correct
    :raises RuntimeError: if conversion fails
    :returns: RDKit molecule
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
    rdkm: Chem.rdchem.Mol
    try:
        try:
            rdkm = xyz2mol(atomic_numbers, coords, charge=charge)[0]  # type: ignore [no-untyped-call]
        except (Exception, ValueError, IndexError):
            rdkm = xyz2mol(atomic_numbers, coords, charge=charge, use_huckel=True)[0]  # type: ignore [no-untyped-call]
    except Exception as e:
        raise SteamrollConversionError("xyz2mol conversion failed!") from e

    return remove_hydrogens(rdkm) if remove_Hs else rdkm
