"""steamroll package."""

import json
import logging
import os
import subprocess
import sys
import tempfile
from collections import Counter
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from rdkit.Geometry import Point3D

from .utils import strip_to_connectivity
from .xyz2mol_tmc.xyz2mol_local import xyz2AC_obabel as xyz2ac_obabel
from .xyz2mol_tmc.xyz2mol_tmc import TRANSITION_METALS_NUM, get_tmc_mol

logger = logging.getLogger(__name__)

# Limit each of the two organic inference attempts; exhaustion never triggers a retry.
_BOND_ORDER_MAX_ITERATIONS = 10_000
_LEGACY_TIMEOUT_SECONDS = 5.0


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
    # RDKit adjusts stereo parity when deleting explicit hydrogen neighbors and
    # retains isotopic hydrogens. Manual atom deletion loses those guarantees.
    return Chem.RemoveHs(molecule, sanitize=False)


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
    base = Chem.MolFromSmiles(smiles)
    if base is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    template_with_h = Chem.AddHs(base)

    n_input = len(atomic_numbers)
    if n_input == template_with_h.GetNumAtoms():
        template = template_with_h
    elif n_input == base.GetNumAtoms():
        template = base
    else:
        raise ValueError(
            f"Atom count mismatch: SMILES has {template_with_h.GetNumAtoms()} atoms "
            f"({base.GetNumAtoms()} heavy), XYZ has {n_input}"
        )

    n = template.GetNumAtoms()

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

    result = Chem.RenumberAtoms(template, inv_match)
    result.RemoveAllConformers()
    result_conf = Chem.Conformer(n)
    for raw_idx in range(n):
        result_conf.SetAtomPosition(raw_idx, Point3D(*xyz_pos[raw_idx].tolist()))
    result.AddConformer(result_conf, assignId=True)
    # Derive stereo from the supplied geometry, then validate any specified stereo
    # against the template. Copying template tags would conceal conflicting poses.
    Chem.RemoveStereochemistry(result)
    Chem.AssignStereochemistryFrom3D(result)
    return result


def _normalize_sulfur(mol: Chem.rdchem.Mol) -> Chem.rdchem.Mol:
    """Copy molecule using consistent sulfoxide and aromatic sulfur representations.

    Preserve Steamroll's existing sulfoxide representation and MMFF typing without
    enumerating charge assignments. Both resonance representations are accepted.
    """
    result = Chem.RWMol(mol)
    for sulfur in result.GetAtoms():
        if (sulfur.GetAtomicNum(), sulfur.GetFormalCharge(), sulfur.GetDegree()) != (16, 1, 3):
            continue
        neighbors = list(sulfur.GetNeighbors())
        if sorted(atom.GetAtomicNum() for atom in neighbors) != [6, 6, 8]:
            continue
        oxygen = next(atom for atom in neighbors if atom.GetAtomicNum() == 8)
        bond = result.GetBondBetweenAtoms(sulfur.GetIdx(), oxygen.GetIdx())
        if (
            oxygen.GetFormalCharge() == -1
            and oxygen.GetDegree() == 1
            and bond.GetBondType() == Chem.BondType.SINGLE
        ):
            sulfur.SetFormalCharge(0)
            oxygen.SetFormalCharge(0)
            bond.SetBondType(Chem.BondType.DOUBLE)
    Chem.SanitizeMol(result)
    # RDKit can place thiazolium charge on sulfur. Its first resonance form moves
    # that charge onto nitrogen, retaining the historical force-field convention.
    if any(
        atom.GetAtomicNum() == 16 and atom.GetIsAromatic() and atom.GetFormalCharge() > 0
        for atom in result.GetAtoms()
    ):
        forms = Chem.ResonanceMolSupplier(result, maxStructs=32)
        if len(forms) and forms[0] is not None:
            normalized = forms[0]
            Chem.SanitizeMol(normalized)
            return normalized
    return result.GetMol()


def _smiles_matches(mol: Chem.rdchem.Mol, smiles: str) -> bool:
    """Check connectivity, isotopes and specified stereo against reference SMILES.

    Unspecified reference stereo accepts coordinate-derived stereochemistry.
    Sulfoxide resonance representations compare equivalently; atom maps are labels.
    """
    ref = Chem.MolFromSmiles(smiles)
    if ref is None:
        return False
    ref = Chem.RemoveHs(_normalize_sulfur(ref))
    got = Chem.RemoveHs(_normalize_sulfur(mol))
    for graph in (ref, got):
        for atom in graph.GetAtoms():
            atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(got, isomericSmiles=False) == Chem.MolToSmiles(
        ref, isomericSmiles=False
    ) and got.HasSubstructMatch(ref, useChirality=True)


def _from_xyz(
    atomic_numbers: list[int], coordinates: list[list[float]], charge: int, use_huckel: bool
) -> Chem.rdchem.Mol:
    """Infer organic bond orders with a finite RDKit iteration budget."""
    mol = Chem.RWMol()
    conf = Chem.Conformer(len(atomic_numbers))
    for i, (number, position) in enumerate(zip(atomic_numbers, coordinates, strict=True)):
        mol.AddAtom(Chem.Atom(number))
        conf.SetAtomPosition(i, Point3D(*position))
    mol.AddConformer(conf)
    rdDetermineBonds.DetermineBonds(
        mol,
        charge=charge,
        useHueckel=use_huckel,
        maxIterations=_BOND_ORDER_MAX_ITERATIONS,
    )
    if Chem.GetFormalCharge(mol) != charge:
        raise ValueError("Inferred molecular charge does not match requested charge")
    return _normalize_sulfur(mol)


def _from_legacy(
    atomic_numbers: list[int], coordinates: list[list[float]], charge: int
) -> Chem.rdchem.Mol | None:
    """Run legacy inference with a shared wall-clock deadline for both attempts."""
    try:
        process = subprocess.run(
            [sys.executable, "-m", "steamroll._legacy"],
            input=json.dumps([atomic_numbers, coordinates, charge]),
            capture_output=True,
            text=True,
            timeout=_LEGACY_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as e:
        raise SteamrollConversionError("Legacy bond-order conversion budget exhausted") from e
    if process.returncode != 0 or not process.stdout.strip():
        logger.debug("Legacy conversion failed: %s", process.stderr)
        return None
    mol = Chem.MolFromMolBlock(json.loads(process.stdout), removeHs=False)
    if mol is None or Chem.GetFormalCharge(mol) != charge:
        return None
    for i, position in enumerate(coordinates):
        mol.GetConformer().SetAtomPosition(i, Point3D(*position))
    return _normalize_sulfur(mol)


def to_rdkit(
    atomic_numbers: Iterable[int],
    coordinates: ArrayLike,
    charge: int = 0,
    remove_Hs: bool = True,
    fail_without_bond_order: bool = True,
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
        fail_without_bond_order: raise on failed bond-order inference by default;
            explicit False permits the existing connectivity-only fallback
        smiles: optional topology template; specified stereochemistry is checked
            against coordinates and unspecified stereo is inferred from coordinates

    Returns:
        RDKit molecule in input atom order, with requested hydrogen removal

    Raises:
        ValueError: input dimensions, elements or coordinates are invalid
        SteamrollConversionError: conversion fails or its inference budget is exhausted
        SteamrollTopologyMismatchError: no candidate matches supplied topology and stereo

    Note:
        Organic XYZ input should include hydrogens. RDKit inference tries geometric
        connectivity, then Hückel connectivity, with 10,000 bond-order iterations per
        attempt. Budget exhaustion is terminal. Legacy inference has a shared five-second
        subprocess deadline and does not retry with charge penalties. These limits do
        not impose a wall-clock deadline on RDKit or specialized metal conversion.
        Carbon-substituted sulfoxides use S=O to preserve existing MMFF typing.
        XYZ alone does not specify isotopes or guarantee a particular spin state.
    """
    atomic_numbers = list(atomic_numbers)
    coordinates = np.asarray(coordinates, dtype=float)

    if coordinates.ndim != 2:
        raise ValueError("`coordinates` needs to be a two-dimensional")
    if coordinates.shape[1] != 3:
        raise ValueError("Coordinates needs to have second dimension with length 3")
    if (n_atoms := len(atomic_numbers)) != (n_coords := len(coordinates)):
        raise ValueError(
            f"Length of atomic numbers ({n_atoms}) doesn't match coordinates ({n_coords})"
        )

    if not atomic_numbers or not np.isfinite(coordinates).all():
        raise ValueError("Coordinates must be nonempty and finite")
    if any(
        not isinstance(number, (int, np.integer)) or not 1 <= number <= 118
        for number in atomic_numbers
    ):
        raise ValueError("Atomic numbers must be integers between 1 and 118")

    atomic_numbers = [int(number) for number in atomic_numbers]
    coords = coordinates.tolist()
    has_tm = any(n in TRANSITION_METALS_NUM for n in atomic_numbers)

    # SMILES-based method: topology from SMILES, positions from XYZ
    if smiles is not None:
        try:
            rdkm = _from_smiles_and_coords(smiles, atomic_numbers, coords)
            if Chem.GetFormalCharge(rdkm) == charge and _smiles_matches(rdkm, smiles):
                rdkm = _normalize_sulfur(rdkm)
                return remove_hydrogens(rdkm) if remove_Hs else rdkm
            logger.debug("SMILES-based conversion produced wrong topology, falling back")
        except (ValueError, RuntimeError) as e:
            logger.debug("SMILES-based conversion failed, falling back: %s", e)

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

    for use_huckel in (False, True):
        try:
            candidate = _from_xyz(atomic_numbers, coords, charge, use_huckel)
            if _topology_ok(candidate):
                rdkm = candidate
                break
        except (ValueError, RuntimeError, IndexError) as e:
            # RDKit translates its C++ iteration exception into RuntimeError.
            if isinstance(e, RuntimeError) and "Max Iterations Exceeded" in str(e):
                raise SteamrollConversionError(
                    "RDKit bond-order iteration budget exhausted; provide SMILES or fix geometry"
                ) from e
            logger.debug("RDKit conversion failed (Hückel=%s): %s", use_huckel, e)

    if rdkm is None:
        candidate = _from_legacy(atomic_numbers, coords, charge)
        if candidate is not None and _topology_ok(candidate):
            rdkm = candidate

    if rdkm is None and fail_without_bond_order:
        error = SteamrollTopologyMismatchError if smiles is not None else SteamrollConversionError
        raise error(
            f"Bond-order inference failed for {len(atomic_numbers)}-atom molecule "
            f"(charge={charge}); provide matching SMILES or fix the geometry"
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
