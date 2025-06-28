# Contributing to Human Voice AI

Thank you for your interest in contributing to Human Voice AI! We appreciate your time and effort in helping us improve this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)
- [Feature Requests](#feature-requests)
- [Documentation](#documentation)

## Code of Conduct

This project and everyone participating in it is governed by our [Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
   ```bash
   git clone https://github.com/your-username/HumanVoiceAI.git
   cd HumanVoiceAI
   ```
3. Set up the development environment (see README.md)
4. Create a new branch for your changes
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Workflow

1. Make your changes
2. Run tests to ensure nothing is broken
   ```bash
   make test
   ```
3. Format your code
   ```bash
   make format
   ```
4. Check for linting errors
   ```bash
   make lint
   ```
5. Commit your changes with a descriptive message
   ```bash
   git commit -m "Add feature: your feature description"
   ```
6. Push to your fork
   ```bash
   git push origin feature/your-feature-name
   ```
7. Open a pull request

## Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **Flake8** for linting
- **Mypy** for static type checking

Run `make format` to automatically format your code before committing.

## Testing

We use `pytest` for testing. To run the test suite:

```bash
make test
```

Please add tests for any new features or bug fixes.

## Pull Request Process

1. Ensure any install or build dependencies are removed before the end of the layer when doing a build.
2. Update the README.md with details of changes to the interface, this includes new environment variables, exposed ports, useful file locations, and container parameters.
3. Increase the version numbers in any examples files and the README.md to the new version that this Pull Request would represent. The versioning scheme we use is [SemVer](http://semver.org/).
4. Your pull request will be reviewed by the maintainers.

## Reporting Issues

When reporting issues, please include the following:

- A clear and descriptive title
- A description of the issue
- Steps to reproduce the issue
- Expected behavior
- Actual behavior
- Screenshots if applicable
- Your environment (OS, Python version, etc.)

## Feature Requests

We welcome feature requests! Please open an issue with:

- A clear and descriptive title
- A description of the feature
- The motivation for the feature
- Any alternative solutions or features you've considered

## Documentation

Good documentation is crucial for any open-source project. If you're adding new features, please update the relevant documentation.

## License

By contributing, you agree that your contributions will be licensed under its MIT License.
