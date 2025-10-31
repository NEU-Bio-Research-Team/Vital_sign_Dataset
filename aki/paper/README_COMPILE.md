# How to Compile Your LaTeX Paper

## Problem
The terminal is using conda's broken pdflatex instead of system TeX Live.

## Solution

### Method 1: Use the Compilation Script (Recommended)
```bash
cd paper
./compile.sh
```

This script **always uses system TeX Live** and avoids conda's broken binaries.

### Method 2: Manual Compilation with Full Paths
```bash
cd paper
/usr/bin/pdflatex -output-directory=./out -interaction=nonstopmode main.tex
/usr/bin/bibtex ./out/main
/usr/bin/pdflatex -output-directory=./out -interaction=nonstopmode main.tex
/usr/bin/pdflatex -output-directory=./out -interaction=nonstopmode main.tex
```

### Method 3: VS Code LaTeX Workshop
VS Code should work fine because `settings.json` points to `/usr/bin/pdflatex`.

Just save any `.tex` file and it should auto-compile.

## Quick Test
```bash
cd paper
./compile.sh
```

Then open: `paper/out/main.pdf`

## Why This Happened
- Conda installs its own TeX Live (incomplete)
- Conda's PATH is first in your shell
- System TeX Live is at `/usr/bin/` (complete installation)
- We need to use system TeX Live explicitly


