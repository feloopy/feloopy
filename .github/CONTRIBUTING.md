# Contributing to FelooPy API

Thank you for considering contributing to FelooPy API! This document outlines the guidelines and workflow for contributing.

## Code of Conduct

Please read and follow our [Code of Conduct](../code_of_conduct.md) before contributing.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git

### Development Setup

1. Fork the repository on GitHub

2. Clone your fork:

```bash
git clone https://github.com/<your-username>/feloopy.git
cd feloopy
```

3. Create a virtual environment and install in development mode:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
pip install pytest  # only needed to run the test suite
```

4. Verify the installation:

```bash
python -c "import feloopy; print(feloopy.__version__)"
```

## How to Contribute

### Reporting Bugs

- Search existing [issues](https://github.com/feloopy/feloopy/issues) to avoid duplicates
- Open a new issue using the **Bug Report** template
- Include a minimal reproducible example
- Provide your Python version and OS

### Suggesting Features

- Open an issue using the **Feature Request** template
- Describe the use case and expected behavior
- Reference any related papers or implementations if applicable

### Submitting Changes

1. Create a new branch from `main`:

```bash
git checkout -b feature/<short-description>
```

2. Make your changes with clear, atomic commits
3. Verify your change works:

```bash
python -c "import feloopy; print(feloopy.__version__)"
```

4. Push to your fork and open a Pull Request

## Development Guidelines

### Code Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions
- Add docstrings to all public functions and classes
- Include type hints where appropriate
- Keep functions focused and concise

### Commit Messages

- Use imperative mood: "Add feature" not "Added feature"
- Keep the subject line under 72 characters
- Reference related issues: `Fix #123`

### Pull Request Checklist

- [ ] Code follows the project's style guidelines
- [ ] Documentation is updated if needed
- [ ] Commit messages are clear and descriptive
- [ ] PR has a descriptive title and summary

## Project Structure

```
feloopy/
├── feloopy/          # Main package source
├── pyproject.toml    # Project metadata and dependencies
└── .github/          # GitHub templates and workflows
```

## Questions?

Open an issue with the **Question** label.
