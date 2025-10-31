# ✅ VS Code LaTeX Workshop - FIXED!

## What I Did

Updated `.vscode/settings.json` to:
1. Use system TeX Live binaries (`/usr/bin/pdflatex`)
2. Set PATH environment to ignore conda's broken TeX Live
3. Disabled auto-build (use manual build with Ctrl+Alt+B)

## How to Use Now

### Option 1: Manual Build in VS Code (Recommended)
1. Open `main.tex` or any section file
2. Press `Ctrl+Alt+B` (or Cmd+Alt+B on Mac)
3. VS Code will compile using system TeX Live
4. PDF will appear in the side panel

### Option 2: Use the Terminal Script
```bash
cd paper
./compile.sh
```

### Option 3: Use Terminal Commands
```bash
cd paper
/usr/bin/pdflatex -output-directory=./out -interaction=nonstopmode main.tex
/usr/bin/bibtex ./out/main
/usr/bin/pdflatex -output-directory=./out -interaction=nonstopmode main.tex
/usr/bin/pdflatex -output-directory=./out -interaction=nonstopmode main.tex
```

## What Changed

- **Tool names**: Changed to `pdflatex-system` and `bibtex-system`
- **PATH override**: `"PATH": "/usr/bin:/bin:/usr/local/bin"` (no conda!)
- **Auto-build**: Disabled (set to "never")
- **Full paths**: `/usr/bin/pdflatex` instead of just `pdflatex`

## Reload VS Code Window

After changing settings, reload VS Code:
1. Press `Ctrl+Shift+P` (or Cmd+Shift+P on Mac)
2. Type "Reload Window"
3. Press Enter

Then try building with `Ctrl+Alt+B`.

## Your PDF

It's already compiled at: `paper/out/main.pdf` (9 pages, 149KB)


