#!/bin/sh

SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR/docker" || { echo "docker directory not found"; exit 1; }
docker compose run tf python "$@"
