#!/usr/bin/env python3
"""Update backup-context.md with current project status."""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def get_git_info():
    """Get git repository information."""
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "-1"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "No git history"


def count_files(directory, pattern):
    """Count files matching pattern."""
    try:
        result = subprocess.run(
            ["find", str(directory), "-type", "f", "-name", pattern],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return len([f for f in result.stdout.strip().split("\n") if f])
    except Exception:
        pass
    return 0


def update_backup_context():
    """Update backup-context.md file."""
    project_root = Path(__file__).parent.parent
    backup_file = project_root / "backup-context.md"

    if not backup_file.exists():
        print("Error: backup-context.md not found")
        return False

    # Read current file
    with open(backup_file, "r", encoding="utf-8") as f:
        content = f.read()

    # Update timestamp
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    content = content.replace(
        "**Last Updated:** 2024-12-27",
        f"**Last Updated:** {current_time}",
    )

    # Update git info if available
    git_info = get_git_info()
    if "No git history" not in git_info:
        # Try to update commit info if section exists
        if "**Commit:**" in content:
            content = content.replace(
                "**Commit:** `7e2064d`",
                f"**Commit:** `{git_info.split()[0] if git_info.split() else 'N/A'}`",
            )

    # Count files
    python_files = count_files(project_root / "src", "*.py")
    yaml_files = count_files(project_root / "configs", "*.yaml")
    md_files = count_files(project_root / "docs", "*.md")

    # Update statistics if section exists
    if "**Python Files:**" in content:
        content = content.replace(
            "**Python Files:** 25 files",
            f"**Python Files:** {python_files} files",
        )

    # Write updated content
    with open(backup_file, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✓ Updated {backup_file}")
    print(f"  Timestamp: {current_time}")
    return True


if __name__ == "__main__":
    update_backup_context()

