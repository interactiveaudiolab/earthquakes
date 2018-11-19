#!/usr/bin/env bash
find . -type f \( -name "*.sh.*" \) -exec xtail -f "$file" {} +
