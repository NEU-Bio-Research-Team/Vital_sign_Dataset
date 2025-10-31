#!/bin/bash
# Wrapper script to force system TeX Live
export PATH=/usr/bin:/bin:/usr/local/bin
unset TEXMFHOME
unset TEXMFVAR
unset TEXMFCONFIG
exec /usr/bin/pdflatex "$@"


