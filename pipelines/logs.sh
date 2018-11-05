#!/usr/bin/env bash
find . -type f \( -name "*.sh.*" \) -exec tail -f "$file" {} +
