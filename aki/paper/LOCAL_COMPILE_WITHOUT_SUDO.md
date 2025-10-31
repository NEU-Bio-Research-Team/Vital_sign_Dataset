# Compile LaTeX Locally Without Admin Access

Since you're in a container without admin privileges, here are solutions to compile LaTeX locally:

## Solution 1: Install TeX Live in Home Directory (Best Option)

You can install TeX Live locally in your home directory without sudo!

### Step 1: Download TeX Live Installer
```bash
cd ~
wget https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz
tar -xzf install-tl-unx.tar.gz
cd install-tl-*/
```

### Step 2: Create Installation Script (No sudo needed)
```bash
# Create installation config
cat > install-tl.conf <<EOF
selected_scheme smallscheme
tlpdbopt_install_docfiles 0
tlpdbopt_install_srcfiles 0
EOF
```

### Step 3: Install to Home Directory
```bash
# This will install to ~/texlive (no sudo needed!)
./install-tl --profile=install-tl.conf

# Add to PATH
echo 'export PATH="$HOME/texlive/2023/bin/$(uname -m)-linux:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Step 4: Verify Installation
```bash
which pdflatex
pdflatex --version
```

---

## Solution 2: Use Docker (If Available)

If Docker is installed but LaTeX is not, use Docker containers:

### Step 1: Create Docker-based LaTeX Compiler
```bash
cd /home/orlab/GIT_PROJECT/Vital_sign_Dataset/paper

# Create helper script
cat > compile.sh <<'EOF'
#!/bin/bash
docker run --rm -v "$PWD":/data -w /data texlive/texlive:latest pdflatex main.tex
EOF

chmod +x compile.sh
```

### Step 2: Compile
```bash
./compile.sh
```

---

## Solution 3: Use VS Code Remote + Overleaf

Since you're in VS Code, you can use Overleaf with VS Code integration:

1. **Install Overleaf Extension in VS Code:**
   - Go to Extensions (Ctrl+Shift+X)
   - Search "Overleaf"
   - Install "Overleaf Sync"

2. **Link to Overleaf:**
   - Create project on Overleaf
   - Use extension to sync locally

3. **Compile on Overleaf, edit locally**

---

## Solution 4: Use Pre-built Binaries

Check if conda/anaconda has TeX:
```bash
conda install -c conda-forge texlive-core
# or
mamba install -c conda-forge texlive-core
```

---

## Solution 5: Minimal LaTeX Installation (User Space)

Install just the minimal packages needed:
```bash
cd ~
mkdir texlive-installer
cd texlive-installer
wget https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz
tar -xzf install-tl-unx.tar.gz
cd install-tl-*

# Minimal installation to ~/texlive
mkdir -p ~/texlive
TEXDIR=~/texlive ./install-tl -profile minimal
```

---

## Solution 6: Use Pandoc as Alternative

If all else fails, convert LaTeX to PDF using Pandoc (might be installed):
```bash
which pandoc
# If available:
pandoc main.tex -o main.pdf
```

---

## Recommended Quick Fix

The **easiest solution** is to use the Docker approach or install minimal TeX Live to home directory:

```bash
# Try Docker first (fastest)
docker run --rm -v "$PWD":/data -w /data texlive/texlive:latest pdflatex main.tex

# If Docker doesn't work, install TeX Live to home directory
cd ~
mkdir -p texlive
cd texlive
wget https://mirror.ctan.org/systems/texlive/tlnet/install-tl-unx.tar.gz
tar -xzf install-tl-unx.tar.gz
cd install-tl-*
./install-tl --no-interaction
```

---

## VS Code Configuration for User-Installed LaTeX

Once LaTeX is installed in home directory, update VS Code settings:

```json
{
    "latex-workshop.latex.tools": [
        {
            "name": "pdflatex",
            "command": "/home/orlab/texlive/bin/x86_64-linux/pdflatex",
            "args": [
                "-synctex=1",
                "-interaction=nonstopmode",
                "-file-line-error",
                "%DOC%"
            ]
        }
    ]
}
```

---

## Check Current Environment

Run these to see what's available:
```bash
which conda
which docker
which mamba
ls ~/texlive 2>/dev/null
ls ~/anaconda3 2>/dev/null
```

---

## Quick Test

After installation, test with a simple file:
```bash
cd /home/orlab/GIT_PROJECT/Vital_sign_Dataset/paper
pdflatex main.tex
```

If compilation works, you'll get `main.pdf`!

---

**Choose the solution that works in your environment!**

