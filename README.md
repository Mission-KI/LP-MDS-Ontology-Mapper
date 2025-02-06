# Extended Data Profile

This repository contains the Extended Data Profile service,
which generates machine and human readable meta information about multi
modal datasets.

# Developer Info

The project's dependencies are managed via pip. Typically, you can install them by running:

```bash
pip install .
```

However, if you require CPU-based PyTorch wheels, you must direct pip to the PyTorch CPU package index. 
You can achieve this by setting the PIP_EXTRA_INDEX_URL environment variable:

```bash
export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cpu
pip install .
```

Alternatively, you can specify the extra index URL directly with pip:

```bash
pip install . --extra-index-url https://download.pytorch.org/whl/cpu
```

## Release Process

To create a release, just push a tag to the repository. The pipeline will then run checks
and create a draft for a release. You can then review it and decide to publish or dump it.

## Scripts

See [Scripts README](scripts/README.md).

## Docker

See [Docker README](docker/README.md).

# Job API

See [Job API README](src/jobapi/README.md).

# Pontus-X CLI

See [Pontus-X CLI README](src/pontusx/README.md).
