#!/bin/bash

set -euo pipefail

pushd $(dirname "$0")/..

DISTRIBUTION_DIR="./dist"
PACKAGE_VERSION_FILE="$(dirname "$0")/src/edps/version.py"

rm -rf ${DISTRIBUTION_DIR} build **/*.egg-info **/__pycache__
mkdir -p ${DISTRIBUTION_DIR}
echo "${VERSION}" > "${DISTRIBUTION_DIR}/VERSION.txt"
echo "__version__ = \"${VERSION}\"\n" > ${PACKAGE_VERSION_FILE}
pip wheel build --no-deps --no-input --no-binary :all: --wheel-dir ${DISTRIBUTION_DIR} .
