# LaTeX Installation Guide for VS Code

The error "spawn latexmk ENOENT" means LaTeX is not installed on your system. Here are installation options:

## Option 1: Install LaTeX Distribution (Recommended)

### For Ubuntu/Debian:
```bash
sudo apt-get update
sudo apt-get install texlive-full
```

### For Fedora/RHEL:
```bash
sudo dnf install texlive-scheme-full
```

### For MacOS (using Homebrew):
```bash
brew install --cask mactex
```

### For Windows:
Download and install MiKTeX from: https://miktex.org/download

After installation, **restart VS Code**.

## Option 2: Use Docker (If Docker is installed)

If you have Docker installed, you can use the LaTeX Docker image. The VS Code config is already set up for this option.

## Option 3: Use Overleaf (Cloud-based)

If local installation is not possible:

1. Go to https://www.overleaf.com
2. Create a new project
3. Upload the `paper/` folder
4. Works instantly, no installation needed!

## Verify Installation

After installing LaTeX, verify it works:

```bash
# Check installation
which pdflatex
which latexmk

# Try compiling manually
cd paper
pdflatex main.tex
```

## VS Code Setup

Once LaTeX is installed:

1. **Install LaTeX Workshop extension** in VS Code:
   - Open Extensions (Ctrl+Shift+X)
   - Search "LaTeX Workshop"
   - Click Install

2. **Open paper/main.tex**

3. **Compile**:
   - Press `Ctrl+Alt+B` (Linux) or `Cmd+Alt+B` (Mac)
   - Or: Command Palette → "LaTeX Workshop: Build with recipe"

4. **View PDF**:
   - PDF opens automatically
   - Use `Ctrl+K V` to toggle PDF view

## Troubleshooting

### Problem: "spawn latexmk ENOENT"
**Solution:** Install LaTeX distribution (see Option 1 above)

### Problem: PDF not opening
**Solution:** Check Output panel for errors, ensure PDF viewer is set correctly

### Problem: SyncTeX not working
**Solution:** Verify `.synctex.gz` file is generated, check VS Code settings

### Problem: Missing packages
**Solution:** Install missing packages:
```bash
sudo apt-get install texlive-science texlive-latex-extra
```

## Current Status

- ❌ LaTeX distribution: Not installed
- ✅ LaTeX Workshop extension: Installed
- ⚠️ Need to install LaTeX (TeX Live, MiKTeX, or MacTeX)

## Quick Start After Installation

Once LaTeX is installed:

```bash
# Navigate to paper directory
cd paper

# Open in VS Code
code main.tex

# Press Ctrl+Alt+B to build
# PDF will appear automatically!
```

## Alternative: Compile Manually

If VS Code still has issues, you can compile manually:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex

# Open PDF
evince main.pdf  # Linux
# or
open main.pdf    # Mac
```

## Questions?

- LaTeX installation: https://www.latex-project.org/get/
- VS Code LaTeX: https://github.com/James-Yu/LaTeX-Workshop
- LaTeX help: https://tex.stackexchange.com/

