#!/bin/bash

set -euo pipefail

pushd $(dirname "$0")/..

scripts/build_wheel.sh
docker build -t beebucket/mds_mapper:latest --file docker/Dockerfile .
