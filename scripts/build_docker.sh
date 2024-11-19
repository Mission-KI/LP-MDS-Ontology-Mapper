# !/bin/bash

set -euo pipefail

THIS_PATH=$(dirname "$0")

${THIS_PATH}/build_wheel.sh
docker build -t beebucket/edps:latest --file ${THIS_PATH}/../docker/Dockerfile .
