#!/bin/bash
# Test if wrapper scripts work correctly

echo "Testing pdflatex-system.sh wrapper..."
echo ""

# Test from paper directory
cd /home/orlab/GIT_PROJECT/Vital_sign_Dataset/paper
echo "Path: $(pwd)"
./pdflatex-system.sh --version 2>&1 | head -1
echo ""

# Test from root directory
cd /home/orlab/GIT_PROJECT/Vital_sign_Dataset
echo "Path: $(pwd)"
./paper/pdflatex-system.sh --version 2>&1 | head -1
echo ""

# Test with absolute path
echo "Absolute path test:"
/home/orlab/GIT_PROJECT/Vital_sign_Dataset/paper/pdflatex-system.sh --version 2>&1 | head -1
echo ""

# Expected output: "pdfTeX 3.141592653-2.6-1.40.24 (TeX Live 2022/Debian)"
# NOT: "pdfTeX 3.141592653-2.6-1.40.25 (TeX Live 2023)"




