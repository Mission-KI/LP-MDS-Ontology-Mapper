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
uv add git+https://github.com/Beebucket/mds_mapper.git

# or with poetry
poetry install git+https://github.com/Beebucket/mds_mapper.git

# or with conda
conda install git+https://github.com/Beebucket/mds_mapper.git

# or with PIP
pip install git+https://github.com/Beebucket/mds_mapper.git
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
git clone https://github.com/Beebucket/mds_mapper.git
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
uv pip install <URL>

# Or using pip
pip install <URL>
```

## Release Process

To create a release, just push a tag to the repository. The pipeline will then run checks
and create a draft for a release. You can then review it and decide to publish or dump it.

## Creating you own service

This is a sample service based on the [Enhanced Dataset Profile Service](https://github.com/Mission-KI/LP-EDPS) (EDPS). It builds upon the [Enhanced Dataset Profile](https://github.com/Mission-KI/LP-EDP) (EDP) schema as a foundational model to implement an Mobility Data Space (MDS) Ontology Mapper, as outlined in the official MDS [documentation](https://github.com/Mobility-Data-Space/mobility-data-space/wiki/MDS-Ontology).

The goal of this example implementation is to illustrate how organizations can define and apply their own Enhanced Dataset Profiles tailored to their specific needs. Using these custom profiles, they can then develop their own EDPS instances to facilitate semantic data integration and interoperability within their ecosystem.

By providing a concrete reference implementation, this service is intended to guide developers and data providers in building scalable, standards-based data services that align with the principles of openness, extensibility, and semantic clarity.

## Scripts

See [Scripts README](scripts/README.md).

## Docker

See [Docker README](docker/README.md).
