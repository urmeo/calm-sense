# Contributing

The sole contributor and maintainer is [urme](https://github.com/urmeo) (`urmeo`).

Follow the [Python setup](docs/reproduction.md#python-setup-and-demo) and [dashboard setup](README.md#run-locally). Run the relevant checks from the repository root:

```bash
python -m black --check src/ tests/ scripts/
python -m pyflakes src/ tests/ scripts/
python -m pyright src/
python -m coverage run --source=src -m unittest discover -s tests
python -m coverage report --fail-under=60
node frontend/tooling.mjs install
node frontend/tooling.mjs test
node frontend/tooling.mjs build
```

GitHub Actions runs these checks for pushes to `main` and pull requests.

- Target `main`. Keep each change focused and add tests for behavior that changes.
- Use short commit messages that describe the change, such as `updated readme`, `updated results`, or `updated code`.
- Fit preprocessing and calibration on training subjects only.
- Keep raw data, caches, and local build files out of commits. Preserve the [recorded results](README.md#recorded-results) unless the change includes a documented rerun.
- Verify dataset download checksums. Document signal units, label mappings, and evaluation conditions.
- Edit frontend dependency and TypeScript settings in `frontend/config/`. The tooling generates the local JSON files. After changing dependencies, run `node frontend/tooling.mjs update` and review both tracked dependency files.
- Keep README `AUTOGEN` markers. Run `python scripts/update_readme_tables.py` after updating calibration or personalization results.

For bug reports, include the command, expected and actual behavior, operating system, and Python or Node.js version. Include the error output when available. Do not attach private recordings or identifiers.

[Report security issues privately](https://github.com/urmeo/calm-sense/security/advisories/new).
