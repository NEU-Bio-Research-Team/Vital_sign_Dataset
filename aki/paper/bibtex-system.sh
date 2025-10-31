#!/bin/bash
# Wrapper script to force system TeX Live
export PATH=/usr/bin:/bin:/usr/local/bin
exec /usr/bin/bibtex "$@"


