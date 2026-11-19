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

To request geometric connectivity without assigning bond orders or charges:

```python
from steamroll import to_connectivity

graph = to_connectivity(atomic_numbers, coordinates)
```

This returns an unsanitized graph with single bonds, retains supplied positions,
and does not infer missing hydrogens, aromaticity, or stereo. It is unsuitable for
force fields or other calculations requiring assigned chemistry. Its boolean
`steamroll_bond_orders_assigned` property is false; successful chemistry conversion
sets this property to true. Explicitly passing `fail_without_bond_order=False` to
`to_rdkit` permits the same graph as a last resort, while inference-budget exhaustion
remains terminal. Call `to_connectivity` directly to skip bond-order inference.
A supplied SMILES always requires matching assigned chemistry, even in permissive mode.


## Credits
This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [jevandezande/uv-cookiecutter](https://github.com/jevandezande/uv-cookiecutter) project template.
