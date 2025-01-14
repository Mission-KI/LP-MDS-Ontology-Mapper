#!/bin/bash

set -euo pipefail

COMPOSE_FILE=$(dirname "$0")/../docker/jobapi/compose.yml

docker compose -f $COMPOSE_FILE up -d
docker compose -f $COMPOSE_FILE logs -f
