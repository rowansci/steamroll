"""Shared RDKit utilities for steamroll."""

from rdkit import Chem


def strip_to_connectivity(mol: Chem.rdchem.Mol) -> Chem.RWMol:
    """Return a copy of mol with only element connectivity preserved.

    Useful for connectivity-only substructure matching where bond order or charge
    differences would otherwise prevent a match.

    Args:
        mol: molecule to strip

    Returns:
        RWMol with all bonds set to SINGLE and chemistry annotations cleared
    """
    rwmol = Chem.RWMol(mol)
    for atom in rwmol.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetIsotope(0)
        atom.SetNumRadicalElectrons(0)
        atom.SetIsAromatic(False)
    for bond in rwmol.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    return rwmol
