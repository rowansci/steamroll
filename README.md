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

Organic XYZ inference expects explicit hydrogens. With `smiles=...`, coordinates may
contain all hydrogens or none, but must contain every heavy atom in the template.
Partial hydrogen inputs and conflicting template charges raise
`SteamrollTopologyMismatchError` before inference. Explicit isotopic hydrogens in
the template require coordinates. Invalid SMILES raise `ValueError`.


## Credits
This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [jevandezande/uv-cookiecutter](https://github.com/jevandezande/uv-cookiecutter) project template.
