# Steamroll

[![License](https://img.shields.io/github/license/rowansci/steamroll)](https://github.com/rowansci/steamroll/blob/master/LICENSE)
[![Powered by: uv](https://img.shields.io/badge/-uv-purple)](https://docs.astral.sh/uv)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typing: ty](https://img.shields.io/badge/typing-ty-EFC621.svg)](https://github.com/astral-sh/ty)
[![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/rowansci/steamroll/test.yml?branch=master&logo=github-actions)](https://github.com/rowansci/steamroll/actions/)
[![Codecov](https://img.shields.io/codecov/c/github/rowansci/steamroll)](https://codecov.io/gh/rowansci/steamroll)
[![PyPI package](https://img.shields.io/pypi/v/steamroll)](https://pypi.org/project/steamroll)

Package for creating RDKit molecules from 3D molecules.

## Usage

Steamroll is simple to use. Simply supply atomic numbers and coordinates (in Å):

```python
from steamroll.steamroll import SteamrollConversionError, to_rdkit

atomic_numbers: list[float] = ...
coordinates: list[float] = ...
charge: int = 0

try:
    rdkit_molecule = to_rdkit(atomic_numbers, coordinates, charge=charge, remove_Hs=True)
except SteamrollConversionError as e:
    raise ValueError("Conversion to RDKit failed!") from e
```


## Credits
This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [jevandezande/uv-cookiecutter](https://github.com/jevandezande/uv-cookiecutter) project template.

## Bond inference and compatibility

`to_rdkit` first tries a supplied SMILES template. Otherwise, ordinary XYZ conversion
uses RDKit `DetermineBonds`, trying geometric connectivity and then Hückel connectivity.
Supply the total molecular charge and include hydrogens when inferring from XYZ alone.
The specialized transition-metal converter is unchanged.

Each RDKit attempt is limited to 10,000 bond-order iterations. If RDKit cannot infer
bonds, the legacy organic converter runs in a subprocess with a shared five-second
deadline covering startup and both connectivity attempts. Legacy sulfur valences try
2 first; exhaustive charge-penalty retries are no longer used. Budget exhaustion raises
`SteamrollConversionError` immediately, without restarting another backend. These bounds
are not an overall wall-clock deadline for RDKit or the specialized metal converter.
RDKit 2025.9.5 or newer is required for the iteration-limit API.

**The default for `fail_without_bond_order` is now `True`.** Callers that intentionally
want the existing connectivity-only fallback can retain that behavior explicitly:

```python
mol = to_rdkit(atomic_numbers, coordinates, charge=charge, fail_without_bond_order=False)
```

This fallback may lack reliable bond orders and formal charges. It is used after
ordinary inference failure, but not after budget exhaustion.

For SMILES-guided conversion, atoms retain template isotope and mapping labels while
coordinates remain in XYZ order. Stereochemistry is assigned from the coordinates;
unspecified reference stereo is accepted, while conflicting specified stereo raises
`SteamrollTopologyMismatchError`. Hydrogen removal preserves stereochemical parity
and retains isotopic hydrogens. XYZ alone cannot recover isotope labels or guarantee
spin multiplicity.

Sulfoxide `S=O` and `S⁺–O⁻` representations are accepted as equivalent. Carbon-substituted
sulfoxides are returned as `S=O` to preserve existing MMFF typing. Aromatic sulfur
cations use RDKit's first resonance form (up to 32 forms), preserving the existing
thiazolium convention. These are representation policies, not claims that other
resonance representations are chemically incorrect.
