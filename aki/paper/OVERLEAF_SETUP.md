# Overleaf Setup Guide (No Installation Required!)

Since you're in a restricted environment without admin privileges, use **Overleaf** - it's the easiest solution!

## Quick Steps to Use Overleaf

### Step 1: Create Overleaf Account
1. Go to: https://www.overleaf.com
2. Sign up for a free account (or sign in)

### Step 2: Create New Project
1. Click **"New Project"**
2. Select **"Blank Project"**
3. Name it: **AXKI Paper**

### Step 3: Upload Your Files

#### Option A: Upload Project Folder (Easiest)
1. In VS Code, zip the `paper/` folder:
   ```bash
   cd /home/orlab/GIT_PROJECT/Vital_sign_Dataset
   zip -r paper.zip paper/
   ```

2. In Overleaf:
   - Click **"Menu"** (top left)
   - Click **"Upload Project"**
   - Upload `paper.zip`
   - Click **"Extract"**

#### Option B: Copy Files Manually
1. Copy content from `main.tex`
2. Create folder structure in Overleaf
3. Paste content into each section file

### Step 4: Edit Your Paper
1. Open `main.tex` in Overleaf
2. Click **"Recompile"** to see PDF
3. PDF appears on the right side automatically!

## Overleaf Features

- ✅ **No installation needed** - Works in browser
- ✅ **Real-time compilation** - PDF updates automatically
- ✅ **Collaboration** - Share with co-authors
- ✅ **Version control** - Automatic history
- ✅ **Rich editor** - Syntax highlighting, autocomplete
- ✅ **No local disk space** - Everything in cloud

## Alternative: Use VS Code Online Editor

If you want to edit locally but compile in Overleaf:

1. Edit files in VS Code normally
2. Sync to Overleaf periodically:
   - Re-upload the folder when ready
   - Or use git integration (advanced)

## Your Project Structure

Your `paper/` folder is already set up correctly:

```
paper/
├── main.tex              ← Main file
├── sections/
│   ├── introduction.tex
│   ├── method.tex
│   ├── results.tex
│   ├── discussion.tex
│   └── conclusion.tex
└── README.md
```

Just upload this entire folder to Overleaf!

## Quick Start Commands

### Upload to Overleaf:
```bash
# Option 1: Create zip
cd ~/GIT_PROJECT/Vital_sign_Dataset
zip -r paper.zip paper/

# Then upload paper.zip to Overleaf
```

### Or copy individual files:
```bash
# View main.tex content
cat paper/main.tex

# Copy this content to Overleaf editor
```

## Working with Overleaf

1. **Edit**: Click any `.tex` file on left side
2. **Compile**: Click "Recompile" button
3. **View PDF**: Automatically shown on right
4. **Download**: Click "Download PDF" when ready

## Sync Back to Local

If you make changes in Overleaf and want them locally:

1. In Overleaf: Click **"Menu"** → **"Download Source"**
2. This downloads a zip
3. Extract to update your local `paper/` folder

## Tips

- Save frequently (Ctrl+S)
- Check compilation logs for errors
- Use Overleaf's real-time preview
- Share project with advisors/collaborators
- No installation needed anywhere!

## Next Steps

1. Upload `paper/` folder to Overleaf
2. Open `main.tex`
3. Start editing - PDF compiles automatically!
4. Download final PDF when ready to submit

