"""Isolated legacy organic inference, bounded by the parent process deadline."""

import contextlib
import json
import sys

from rdkit import Chem

from .xyz2mol.xyz2mol import xyz2mol


def main() -> None:
    """Try legacy connectivity methods without exhaustive charge-penalty retries."""
    numbers, coordinates, charge = json.loads(sys.stdin.read())
    for use_huckel in (False, True):
        try:
            with contextlib.redirect_stdout(sys.stderr):
                mol = xyz2mol(numbers, coordinates, charge=charge, use_huckel=use_huckel)[0]
                Chem.SanitizeMol(mol)
            if Chem.GetFormalCharge(mol) == charge:
                print(json.dumps(Chem.MolToMolBlock(mol)))
                return
        except (ValueError, RuntimeError, IndexError, KeyError):
            continue


if __name__ == "__main__":
    main()
