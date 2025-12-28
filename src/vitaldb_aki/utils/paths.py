"""Path management utilities."""

from pathlib import Path
from typing import Dict

from ..config import Config


def get_paths(config: Config) -> Dict[str, Path]:
    """Get all artifact paths from config.

    Args:
        config: Configuration object.

    Returns:
        Dictionary of path names to Path objects.
    """
    return {
        "artifacts_dir": Path(config.artifacts_dir),
        "cache_dir": Path(config.cache_dir),
        "tables_dir": Path(config.tables_dir),
        "scalers_dir": Path(config.scalers_dir),
        "plots_dir": Path(config.plots_dir),
        "models_dir": Path(config.artifacts_dir) / "models",
        "results_dir": Path(config.artifacts_dir) / "results",
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

