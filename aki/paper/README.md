# AXKI Paper - LaTeX Project

This folder contains a complete LaTeX project for writing the AXKI paper. The project is configured to compile **locally without requiring admin access**.

## ✨ Features

- ✅ **No Admin Access Required** - Uses local build directory (`./out`)
- ✅ **Auto-compile on Save** - Automatically builds when you save files
- ✅ **PDF Preview in Tab** - Opens PDF side-by-side with source
- ✅ **SyncTeX** - Click in PDF to jump to source, click in source to jump to PDF
- ✅ **Complete Project Structure** - Introduction, Method, Results, Discussion, Conclusion
- ✅ **Bibliography Management** - Uses biblatex with references.bib

## 📁 Project Structure

```
paper/
├── main.tex              # Main LaTeX file (entry point)
├── sections/             # Section files
│   ├── introduction.tex
│   ├── method.tex
│   ├── results.tex
│   ├── discussion.tex
│   └── conclusion.tex
├── images/               # Images directory (with placeholder)
├── references.bib        # Bibliography file
├── out/                  # Build output (auto-generated, git-ignored)
├── .vscode/settings.json # LaTeX Workshop configuration
└── README.md             # This file
```

## 🚀 Quick Start

### Prerequisites

- **VS Code or CursorAI** with **LaTeX Workshop extension** installed ✅
- **TeX Live or MikTeX** installed (required for compilation)
- **Anaconda/Conda** with `texlive-core` package

### How to Build and Preview

1. **Open `paper/main.tex`** in VS Code/CursorAI
2. **Press `Ctrl+Alt+B`** (Windows/Linux) or `Cmd+Option+B` (Mac)
   - Or: Command Palette → `LaTeX Workshop: Build with recipe`
3. **View PDF**: 
   - Automatically opens in a new tab
   - Or: Press `Ctrl+K V` to toggle preview
4. **Edit and save**: PDF updates automatically!

## 🔧 How This Works (No Admin Required)

### Key Configuration

The project uses a **local build directory** (`./out`) instead of system directories:

```json
{
    "latex-workshop.latex.outDir": "./out",  // Local build folder
    "latex-workshop.latex.autoBuild.run": "onSave"  // Auto-compile
}
```

This means:
- ✅ **No system-level installation** required
- ✅ **No admin privileges** needed
- ✅ **All build files** stored locally in `./out/`
- ✅ **Works in containers** and restricted environments

### Build Process

When you build (`Ctrl+Alt+B`):

1. **pdflatex** compiles `main.tex` → creates `main.aux`
2. **bibtex** processes `references.bib` → creates `main.bbl`
3. **pdflatex** runs again to incorporate citations
4. **pdflatex** runs final time for cross-references
5. **Output**: `out/main.pdf` appears in new tab!

## 📝 Adding Content

### Add New Section

1. Create `sections/newsection.tex`
2. Add to `main.tex`: `\input{sections/newsection}`
3. Save - PDF updates automatically!

### Add Figures

1. Place image in `images/` folder (PNG, PDF, JPEG)
2. Add to section file:
   ```latex
   \begin{figure}[H]
       \centering
       \includegraphics[width=0.8\textwidth]{images/your-figure.png}
       \caption{Your caption}
       \label{fig:yourlabel}
   \end{figure}
   ```

### Add Citations

1. Edit `references.bib` (add entry)
2. Cite in text: `\cite{key}`
3. Save - bibliography updates automatically!

## 🎯 LaTeX Workshop Commands

| Keybinding | Action |
|------------|--------|
| `Ctrl+Alt+B` | Build LaTeX project |
| `Ctrl+K V` | Toggle PDF preview |
| `Ctrl+Click` (in PDF) | Jump to source |
| `Ctrl+Click` (in source) | Jump to PDF |
| `Ctrl+Shift+P` → "Clean" | Clean auxiliary files |

## ✅ Current Status

- ✅ Project structure complete
- ✅ All sections written
- ✅ References configured
- ✅ VS Code settings ready
- ✅ Local build directory set up
- ⚠️ Requires TeX Live installation

## 🐛 Troubleshooting

### "spawn latexmk ENOENT"

**Solution**: TeX Live needs to be installed. Since admin access is restricted:
1. Use conda: `conda install -c conda-forge texlive-core`
2. Or: Use Overleaf (upload this folder to overleaf.com)

### PDF not updating

**Solution**: 
1. Check Output panel for errors
2. Run "LaTeX Workshop: Clean up auxiliary files"
3. Rebuild with `Ctrl+Alt+B`

### Citation errors

**Solution**: Install biber:
```bash
conda install -c conda-forge biber
```

### Missing packages

**Solution**: Install via conda:
```bash
conda install -c conda-forge texlive-full
```

## 📖 Additional Resources

- **LaTeX Workshop Docs**: https://github.com/James-Yu/LaTeX-Workshop
- **Overleaf Alternative**: https://www.overleaf.com (upload this folder)
- **Installation Guide**: See `INSTALLATION.md`

## 🎉 Success!

When everything works, you'll see:
- ✅ PDF opens automatically when you build
- ✅ Click in PDF → jumps to source line
- ✅ Edit code → PDF updates on save
- ✅ All files compile locally without admin access

**Happy Writing!** 📄
