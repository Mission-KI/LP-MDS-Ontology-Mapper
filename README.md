# MDS Mapper Service

The Mobility Data Space Mapper Service maps Extended Dataset Profiles into
the mobility data space specific ontology.

# User Info

The MDS Mapper service is shipped as a docker image. The path
of the docker image is always annotated on the release notes.
You can also [create the docker image locally by yourself](#scripts).

You can run the service by calling:

```bash
./scripts/start_jobapi.sh
```

# Developer Info

## Library Installation

You can install this package in your project with the UV package manager,
but any python package manager should work. We advice to always specify a
tag, so your builds stay reproducible:

```bash
# With UV package manager
uv add git+https://github.com/Mission-KI/LP-EDP

# or with poetry
poetry install git+https://github.com/Mission-KI/LP-EDP

# or with conda
conda install git+https://github.com/Mission-KI/LP-EDP

# or with PIP
pip install git+https://github.com/Mission-KI/LP-EDP
```

## Development setup

The project's dependencies are managed via the uv package manager. Even though, the pyproject.toml is
compliant to other package managers as well, but you might need to add a reference to the pytorch
package sources when installing. We advise using the uv package manager for increased speed and
better support of the pyproject.toml.

[Here you can find information on how to install uv.](https://docs.astral.sh/uv/getting-started/installation).
We strongly advise NOT to install uv in the hosts python environment via pip!

After you installed uv, you can install this repository as follows. The command will create a ".venv" directory
inside your repository:

```sh
git clone https://github.com/Mission-KI/LP-EDP
cd mds_mapper
uv sync --all-extras
```

For linting we use pre-commit. You can enable checking all commits by running this command inside your repository
directory:
```bash
uv run pre-commit install
```

We utilize pytest for the unit tests. You can run the test either through your IDE's test section, or by calling:
```sh
uv run pytest .
```

You can instead install the package directly from the repository, which will result in installing the package and
its entrypoints into your environment:
```sh
# Using UV
uv pip install https://github.com/Mission-KI/LP-EDP

# Or using pip
pip install https://github.com/Mission-KI/LP-EDP
```

## Release Process

To create a release, just push a tag to the repository. The pipeline will then run checks
and create a draft for a release. You can then review it and decide to publish or dump it.

## Creating you own service

You can create your own service by forking this repository.
Strip the components you do not need. Then start adding in your own functionality.

## Scripts

See [Scripts README](scripts/README.md).

## Docker

See [Docker README](docker/README.md).
