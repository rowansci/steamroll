# Steamroll

[![License](https://img.shields.io/github/license/rowansci/steamroll)](https://github.com/rowansci/steamroll/blob/master/LICENSE)
[![Powered by: Pixi](https://img.shields.io/badge/Powered_by-Pixi-facc15)](https://pixi.sh)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
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
This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [jevandezande/pixi-cookiecutter](https://github.com/jevandezande/pixi-cookiecutter) project template.
