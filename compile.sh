#!/usr/bin/env bash
pushd docs
make html
make markdown
mv build/markdown/index.md ../README.md
popd