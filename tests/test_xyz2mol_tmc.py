"""Tests for lanthanide and actinide support in xyz2mol_tmc."""

from pathlib import Path

import pytest
from rdkit import Chem

from steamroll.xyz2mol_tmc.xyz2mol_tmc import (
    ALLOWED_OXIDATION_STATES,
    TRANSITION_METALS,
    TRANSITION_METALS_NUM,
    MetalNon_Hg,
    get_tmc_mol,
)

HERE = Path(__file__).parent
DATA_DIR = HERE / "data"

# ---------------------------------------------------------------------------
# Smoke tests: lookup-table completeness
# ---------------------------------------------------------------------------

LANTHANIDE_NUMS = list(range(57, 72))   # La(57) through Lu(71)
ACTINIDE_NUMS   = list(range(89, 104))  # Ac(89) through Lr(103)


def test_lanthanide_nums_in_transition_metals_num():
    for n in LANTHANIDE_NUMS:
        assert n in TRANSITION_METALS_NUM, f"Atomic number {n} missing from TRANSITION_METALS_NUM"


def test_actinide_nums_in_transition_metals_num():
    for n in ACTINIDE_NUMS:
        assert n in TRANSITION_METALS_NUM, f"Atomic number {n} missing from TRANSITION_METALS_NUM"


def test_lanthanide_symbols_in_allowed_oxidation_states():
    lanthanide_symbols = ["La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu",
                          "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"]
    for sym in lanthanide_symbols:
        assert sym in ALLOWED_OXIDATION_STATES, f"{sym} missing from ALLOWED_OXIDATION_STATES"


def test_actinide_symbols_in_allowed_oxidation_states():
    actinide_symbols = ["Ac", "Th", "Pa", "U", "Np", "Pu", "Am",
                        "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"]
    for sym in actinide_symbols:
        assert sym in ALLOWED_OXIDATION_STATES, f"{sym} missing from ALLOWED_OXIDATION_STATES"


def test_lanthanide_symbols_in_transition_metals():
    lanthanide_symbols = ["Ce", "Pr", "Nd", "Pm", "Sm", "Eu",
                          "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb"]
    for sym in lanthanide_symbols:
        assert sym in TRANSITION_METALS, f"{sym} missing from TRANSITION_METALS"


def test_actinide_symbols_in_transition_metals():
    actinide_symbols = ["Ac", "Th", "Pa", "U", "Np", "Pu", "Am",
                        "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"]
    for sym in actinide_symbols:
        assert sym in TRANSITION_METALS, f"{sym} missing from TRANSITION_METALS"


# ---------------------------------------------------------------------------
# SMARTS validity
# ---------------------------------------------------------------------------

def test_metal_non_hg_smarts_parses():
    mol = Chem.MolFromSmarts(MetalNon_Hg)
    assert mol is not None, "MetalNon_Hg SMARTS failed to parse"


def test_metal_non_hg_contains_lanthanides():
    for n in range(58, 71):
        assert f"#{n}" in MetalNon_Hg, f"#{n} (lanthanide) missing from MetalNon_Hg SMARTS"


def test_metal_non_hg_contains_actinides():
    for n in range(89, 104):
        assert f"#{n}" in MetalNon_Hg, f"#{n} (actinide) missing from MetalNon_Hg SMARTS"


# ---------------------------------------------------------------------------
# End-to-end: lanthanide (Ce with 18-crown-6)
# ---------------------------------------------------------------------------

def test_ce_crown_ether_end_to_end():
    """Ce(+1) 18-crown-6 complex should produce a valid, sanitizable mol."""
    xyz_file = str(DATA_DIR / "Ce_18-crown-6.xyz")
    mol = get_tmc_mol(xyz_file, overall_charge=1)

    assert mol is not None, "get_tmc_mol returned None for Ce 18-crown-6"
    Chem.SanitizeMol(mol)  # must not raise

    # Find the Ce atom and verify it has a formal charge (= oxidation state)
    ce_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 58]
    assert len(ce_atoms) == 1, "Expected exactly one Ce atom"
    ce_charge = ce_atoms[0].GetFormalCharge()
    assert ce_charge == 1, f"Expected Ce formal charge +1, got {ce_charge}"


# ---------------------------------------------------------------------------
# End-to-end: actinide ([UCl6]2-)
# ---------------------------------------------------------------------------

def test_ucl6_end_to_end():
    """[UCl6]2- (octahedral U(IV)) should produce a valid, sanitizable mol."""
    xyz_file = str(DATA_DIR / "UCl6.xyz")
    mol = get_tmc_mol(xyz_file, overall_charge=-2)

    assert mol is not None, "get_tmc_mol returned None for UCl6"
    Chem.SanitizeMol(mol)  # must not raise

    # U should have +4 formal charge: overall -2, six Cl- = -6, so U = +4
    u_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 92]
    assert len(u_atoms) == 1, "Expected exactly one U atom"
    u_charge = u_atoms[0].GetFormalCharge()
    assert u_charge == 4, f"Expected U formal charge +4, got {u_charge}"


# ---------------------------------------------------------------------------
# Regression: d-block metal still works (ferrocene)
# ---------------------------------------------------------------------------

def test_ferrocene_regression():
    """Ferrocene should still produce a valid mol with Fe(+2)."""
    xyz_file = str(DATA_DIR / "ferrocene.xyz")
    mol = get_tmc_mol(xyz_file, overall_charge=0)

    assert mol is not None, "get_tmc_mol returned None for ferrocene"
    Chem.SanitizeMol(mol)

    fe_atoms = [a for a in mol.GetAtoms() if a.GetAtomicNum() == 26]
    assert len(fe_atoms) == 1, "Expected exactly one Fe atom"
    fe_charge = fe_atoms[0].GetFormalCharge()
    assert fe_charge == 2, f"Expected Fe formal charge +2, got {fe_charge}"
