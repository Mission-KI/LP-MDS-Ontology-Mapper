#!/bin/bash

set -euo pipefail

pushd $(dirname "$0")/..

scripts/build_wheel.sh
docker build -t beebucket/edps:latest --file docker/Dockerfile .
