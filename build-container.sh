#!/bin/bash
if [ -z "$1" ]; then
  echo "error: you must specify the version"
  echo "usage: ./build-container <version>"
  exit 1
fi
VERSION=$1
R_USER=khaller
docker build -f docker/Dockerfile . -t "${R_USER}/kgrec:${VERSION}"
