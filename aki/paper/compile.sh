#!/bin/bash
# LaTeX Compilation Script
# Always uses system TeX Live binaries

cd /home/orlab/GIT_PROJECT/Vital_sign_Dataset/paper

echo "🔨 Compiling LaTeX document..."
echo ""

# Clean previous output
rm -f ./out/*.aux ./out/*.bbl ./out/*.blg ./out/*.log ./out/*.out ./out/*.toc

echo "Step 1/3: First pdflatex pass..."
/usr/bin/pdflatex -output-directory=./out -interaction=nonstopmode main.tex > /dev/null 2>&1

echo "Step 2/3: Running BibTeX..."
/usr/bin/bibtex ./out/main > /dev/null 2>&1

echo "Step 3/3: Final pdflatex passes..."
/usr/bin/pdflatex -output-directory=./out -interaction=nonstopmode main.tex > /dev/null 2>&1
/usr/bin/pdflatex -output-directory=./out -interaction=nonstopmode main.tex > /dev/null 2>&1

echo ""
echo "✅ Compilation complete!"
echo "📄 PDF output: ./out/main.pdf"
echo ""
ls -lh ./out/main.pdf


