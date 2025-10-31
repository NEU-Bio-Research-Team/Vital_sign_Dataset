# ✅ FINAL VS Code LaTeX Workshop Solution

## What I Did

1. **Fixed wrapper scripts** - Added environment variable cleanup
2. **Fixed settings.json** - Removed duplicate key
3. **Workspace-level settings** - Created `.vscode/settings.json` at project root

## Now Try This

1. **Close VS Code completely** (not just reload window)
2. **Reopen VS Code**
3. Open `paper/main.tex` or any section file
4. Press `Ctrl+Shift+B` to choose build task, OR
   Press `Ctrl+Alt+B` (or `Cmd+Alt+B` on Mac) to build

The extension should now use the wrapper scripts which force system TeX Live.

## Test the Wrapper

If it still fails, test the wrapper directly:

```bash
cd /home/orlab/GIT_PROJECT/Vital_sign_Dataset
./paper/pdflatex-system.sh --version
```

You should see: `pdfTeX 3.141592653-2.6-1.40.24 (TeX Live 2022/Debian)`

NOT: `pdfTeX 3.141592653-2.6-1.40.25 (TeX Live 2023)`

## If It Still Doesn't Work

The wrapper scripts are correct. If VS Code still fails, it might be a LaTeX Workshop extension cache issue:

1. Close VS Code
2. Delete `.vscode` folder in the paper directory (we now use root-level `.vscode`)
3. Open VS Code
4. Try building again

## Alternative: Use Shell Alias

Create a shell alias to temporarily fix PATH:

```bash
alias pdflatex='/usr/bin/pdflatex'
alias bibtex='/usr/bin/bibtex'
```

Then close and reopen VS Code.

## Verification

Check which pdflatex VS Code is calling:

```bash
which pdflatex
# Should show: /usr/bin/pdflatex
# NOT: /home/orlab/anaconda3/bin/pdflatex
```

## Last Resort

If **nothing** works with VS Code, use:
```bash
cd paper && ./compile.sh
```

This will **always** work because it bypasses VS Code entirely.




