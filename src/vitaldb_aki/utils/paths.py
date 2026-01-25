"""Path management utilities."""

from pathlib import Path
from typing import Dict, Union

from ..config import Config


def _project_root() -> Path:
    """Return the Vital_sign_Dataset project root.

    This makes relative artifact paths deterministic even if scripts are run
    from a different working directory.
    """
    # .../Vital_sign_Dataset/src/vitaldb_aki/utils/paths.py -> .../Vital_sign_Dataset
    return Path(__file__).resolve().parents[3]


def _resolve_under_root(p: Union[str, Path]) -> Path:
    """Resolve a path under project root if it's relative."""
    pp = Path(p)
    if pp.is_absolute():
        return pp
    return _project_root() / pp


def get_paths(config: Config) -> Dict[str, Path]:
    """Get all artifact paths from config.

    Args:
        config: Configuration object.

    Returns:
        Dictionary of path names to Path objects.
    """
    artifacts_dir = _resolve_under_root(config.artifacts_dir)
    return {
        "artifacts_dir": artifacts_dir,
        "cache_dir": _resolve_under_root(config.cache_dir),
        "tables_dir": _resolve_under_root(config.tables_dir),
        "scalers_dir": _resolve_under_root(config.scalers_dir),
        "plots_dir": _resolve_under_root(config.plots_dir),
        "models_dir": artifacts_dir / "models",
        "results_dir": artifacts_dir / "results",
    }


def setup_directories(config: Config) -> Dict[str, Path]:
    """Create all necessary directories.

    Args:
        config: Configuration object.

    Returns:
        Dictionary of path names to Path objects.
    """
    paths = get_paths(config)
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths

