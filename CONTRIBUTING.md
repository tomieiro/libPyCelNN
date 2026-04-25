# Contributing

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

For optional image, SciPy, and plotting features:

```bash
pip install -e .[all]
```

## Development workflow

1. Create a feature branch.
2. Make focused changes with tests.
3. Run `pytest`.
4. Run `ruff check .`.
5. Open a pull request with a concise description of the change and its motivation.

## Style guidelines

- Keep the core package independent from optional image and plotting dependencies.
- Prefer readable numerical code over compact but opaque implementations.
- Add docstrings to public classes and functions.
- Raise `celnn` exceptions with explicit, actionable messages.

## Testing

The test suite is written with `pytest`. Optional-dependency tests skip cleanly when Pillow or SciPy are unavailable.

```bash
PYTHONPATH=src pytest
```
