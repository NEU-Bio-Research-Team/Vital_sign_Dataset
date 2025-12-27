"""Configuration management for VitalDB AKI prediction."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


@dataclass(frozen=True)
class Config:
    """Configuration dataclass for the VitalDB AKI prediction pipeline."""

    # Signals
    signals: Tuple[str, ...] = ("ART_MBP", "CVP", "NEPI_RATE", "PLETH_HR", "PLETH_SPO2")
    required_signals: Tuple[str, ...] = ("ART_MBP", "PLETH_HR", "PLETH_SPO2")
    include_optional_signals: bool = True

    # Resampling
    fs_hz: float = 1.0
    max_len_sec: int = 4 * 3600
    min_len_sec: int = 10 * 60
    min_obs_points_per_channel: int = 60

    # Feature time window / cutoff (anti-leakage)
    cutoff_mode: str = "early_intraop"  # "early_intraop" or "preop"
    t_cut_sec: int = 60 * 60
    preop_window_sec: int = 60 * 60

    # AKI label windows (seconds)
    baseline_window_sec: int = 30 * 24 * 3600
    postop_window_sec: int = 7 * 24 * 3600

    # CV / splits
    n_splits: int = 5
    random_state: int = 42

    # Performance
    max_cases_to_cache: Optional[int] = None
    n_threads: int = 4

    # IO
    api_base: str = "https://api.vitaldb.net"
    artifacts_dir: str = "artifacts/demo_5signals"
    cache_dir: str = "artifacts/demo_5signals/cache_npz"
    tables_dir: str = "artifacts/demo_5signals/tables"
    scalers_dir: str = "artifacts/demo_5signals/scalers"
    plots_dir: str = "artifacts/demo_5signals/plots"

    # Training hyperparameters
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    monitor: str = "pr_auc"  # "pr_auc" or "roc_auc"
    device: str = "cuda"  # "cuda" or "cpu"

    # Model architecture
    model_hidden_dim: int = 64
    model_num_layers: int = 2
    model_dropout: float = 0.2

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> Config:
        """Create config from dictionary."""
        # Convert lists back to tuples for signals
        if "signals" in d and isinstance(d["signals"], list):
            d["signals"] = tuple(d["signals"])
        if "required_signals" in d and isinstance(d["required_signals"], list):
            d["required_signals"] = tuple(d["required_signals"])
        return cls(**d)

    def save(self, path: Path) -> None:
        """Save config to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> Config:
        """Load config from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls.from_dict(d)

    @classmethod
    def from_yaml(cls, path: Path) -> Config:
        """Load config from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)

    def update_paths_for_experiment(self, experiment_name: str) -> Config:
        """Update artifact paths to use experiment name."""
        base_dir = f"artifacts/{experiment_name}"
        d = self.to_dict()
        d.update({
            "artifacts_dir": base_dir,
            "cache_dir": f"{base_dir}/cache_npz",
            "tables_dir": f"{base_dir}/tables",
            "scalers_dir": f"{base_dir}/scalers",
            "plots_dir": f"{base_dir}/plots",
        })
        return Config.from_dict(d)


def load_config(config_path: Optional[Path] = None, experiment_name: Optional[str] = None) -> Config:
    """Load configuration from file or use defaults.

    Args:
        config_path: Path to YAML config file. If None, uses default config.
        experiment_name: Experiment name to update artifact paths.

    Returns:
        Config object.
    """
    if config_path is not None and config_path.exists():
        config = Config.from_yaml(config_path)
    else:
        config = Config()

    if experiment_name is not None:
        config = config.update_paths_for_experiment(experiment_name)

    return config

