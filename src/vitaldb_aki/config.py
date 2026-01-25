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
    # Required: ART_MBP, PLETH_HR, PLETH_SPO2
    # Optional: ART_SBP, ART_DBP, HR, ETCO2
    # NOTE: Internally, `signals` MUST include required signals (tensor channels).
    # YAML configs in this repo may list only optional signals under `signals`; loading will auto-merge.
    signals: Tuple[str, ...] = (
        "ART_MBP",
        "PLETH_HR",
        "PLETH_SPO2",
        "ART_SBP",
        "ART_DBP",
        "HR",
        "ETCO2",
    )
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

    # Fold stratification
    fold_stratify_use_n_postop_labs: bool = False
    # If True, additionally stratify on whether creatinine change is near the AKI threshold.
    # This helps avoid a single fold accumulating many borderline-label cases.
    fold_stratify_use_cr_margin_bin: bool = True
    # Relative margin threshold for the creatinine near-threshold bin.
    # A case is considered "near threshold" when:
    #   abs(postop_max_cr - 1.5 * baseline_cr) / baseline_cr < fold_cr_margin_rel_threshold
    fold_cr_margin_rel_threshold: float = 0.10

    # Performance
    max_cases_to_cache: Optional[int] = None
    n_threads: int = 4

    # IO
    api_base: str = "https://api.vitaldb.net"
    artifacts_dir: str = "artifacts/new_optional_exp"
    cache_dir: str = "artifacts/new_optional_exp/cache_npz"
    tables_dir: str = "artifacts/new_optional_exp/tables"
    scalers_dir: str = "artifacts/new_optional_exp/scalers"
    plots_dir: str = "artifacts/new_optional_exp/plots"

    # Training hyperparameters
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    monitor: str = "pr_auc"  # "pr_auc" or "roc_auc"
    device: str = "cuda"  # "cuda" or "cpu"

    # Model architecture
    model_hidden_dim: int = 64
    model_num_layers: int = 2
    model_dropout: float = 0.2
    # Architecture knobs for hybrid models
    model_tcn_levels: int = 4
    model_rnn_layers: int = 4
    model_cell_type: str = "lstm"  # "lstm" or "gru"
    model_use_attention: bool = True
    model_kernel_size: int = 3
    # WaveNet / WaveNetRNN specific
    model_num_stacks: int = 3
    model_num_layers_per_stack: int = 10
    # Attention/Transformer specific
    model_num_heads: int = 4
    model_use_multiscale: bool = True
    model_max_len: int = 3600

    # Optional lightweight TCN branch (deprecated; kept for backward compatibility)
    model_v2_use_tcn_branch: bool = False
    model_v2_tcn_branch_levels: int = 2
    model_v2_tcn_branch_kernel_size: int = 3

    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> Config:
        """Create config from dictionary."""
        # Convert lists back to tuples
        if "signals" in d and isinstance(d["signals"], list):
            d["signals"] = tuple(d["signals"])
        if "required_signals" in d and isinstance(d["required_signals"], list):
            d["required_signals"] = tuple(d["required_signals"])

        # Ensure required_signals are always included in signals (channels)
        required = tuple(d.get("required_signals") or ())
        signals = tuple(d.get("signals") or ())
        if required:
            merged: List[str] = list(required)
            for s in signals:
                if s not in required:
                    merged.append(s)
            d["signals"] = tuple(merged)
        elif signals:
            # No required list provided; keep signals as-is.
            d["signals"] = signals
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


def _project_root() -> Path:
    """Return the Vital_sign_Dataset project root.

    Keep artifact/config resolution deterministic even if scripts are run
    from a different working directory.
    """
    # .../Vital_sign_Dataset/src/vitaldb_aki/config.py -> .../Vital_sign_Dataset
    return Path(__file__).resolve().parents[2]


def load_config(
    config_path: Optional[Path] = None,
    experiment_name: Optional[str] = None,
    *,
    prefer_saved_config: bool = True,
) -> Config:
    """Load configuration from file or use defaults.

    Args:
        config_path: Path to YAML config file. If None, uses default config.
        experiment_name: Experiment name to update artifact paths.

    Returns:
        Config object.
    """
    # Priority order:
    # 1) Explicit YAML passed by user
    # 2) Existing artifacts/<experiment>/config.json (keeps train/eval consistent with preprocess)
    # 3) Built-in defaults
    if config_path is not None and config_path.exists():
        config = Config.from_yaml(config_path)
        if experiment_name is not None:
            config = config.update_paths_for_experiment(experiment_name)
        return config

    if prefer_saved_config and experiment_name is not None:
        # Try to load the config that was saved during preprocessing
        candidate = _project_root() / "artifacts" / experiment_name / "config.json"
        if candidate.exists():
            config = Config.load(candidate)
            # Ensure paths match the requested experiment_name (safe if already correct)
            return config.update_paths_for_experiment(experiment_name)

    config = Config()
    if experiment_name is not None:
        config = config.update_paths_for_experiment(experiment_name)
    return config

