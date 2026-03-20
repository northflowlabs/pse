# Contributing to Northflow PSE

Thank you for your interest in contributing to PSE. We welcome contributions from the Earth observation, climate science, and Python communities.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/pse.git`
3. Install dependencies: `pip install -r requirements.txt && pip install pytest-cov ruff python-dotenv`
4. Copy environment template: `cp .env.example .env`
5. Create a branch: `git checkout -b feature/your-feature`
6. Make your changes
7. Run tests: `pytest -m "not integration" --tb=short`
8. Run the linter: `ruff check .`
9. Push and open a Pull Request

## Code Style

- We use [Ruff](https://github.com/astral-sh/ruff) for linting and formatting
- Type hints are required on all public functions
- Docstrings follow NumPy style
- All new connectors must include unit tests with mocked HTTP responses

## Adding a New Connector

This is the highest-impact contribution you can make. See [../docs/connectors.md](../docs/connectors.md) for the step-by-step guide.

1. Create `connectors/your_source.py` implementing `BaseConnector`
2. Add unit tests in `tests/pse/test_connectors/test_your_source.py`
3. Add an integration test marked with `@pytest.mark.integration`
4. Document the source in `docs/data-sources.md`
5. Register the connector in `api/main.py`

## Reporting Issues

Use the issue templates:

- **Bug Report** — Something is broken
- **Feature Request** — Suggest an enhancement
- **New Connector** — Propose a new data source

## Code of Conduct

We follow the [Contributor Covenant](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). Be kind, constructive, and professional.
