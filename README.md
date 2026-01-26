# VitalDB AKI Prediction

A machine learning pipeline for predicting Acute Kidney Injury (AKI) from intraoperative vital signs using data from VitalDB.

## Overview

This project implements a deep learning pipeline to predict AKI occurrence after surgery using the first 60 minutes of intraoperative vital signs. AKI is defined as postoperative creatinine ≥ 1.5× baseline creatinine.

### Key Features

- **7 Vital Signs**: PLETH_SPO2, PLETH_HR, ART_MBP, ART_SBP, ART_DBP, HR, ETCO2
- **4 Model Architectures**: TCN, GRU, LSTM, BiLSTM, MLP, WaveNet, Dilated Convolutional Network, Dilated Recurrent Network, SynerT
- **5-Fold Cross-Validation**: Stratified splits with reproducible results
- **Missingness-Aware Processing**: Explicit mask channels for missing data
- **Anti-Leakage Design**: Feature window cutoff to prevent label leakage
- **Automatic Checkpointing**: Skip completed steps on re-runs

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended) or CPU
- Conda (recommended) or pip

### Option 1: Conda Environment (Recommended)

```bash
# Create minimal conda environment (fast - only Python)
conda env create -f environment.yml

# Activate environment
conda activate vitaldb_aki

# Install PyTorch with CUDA support (choose appropriate CUDA version)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU-only (no GPU):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies from requirements.txt
pip install -r requirements.txt

# Verify CUDA is available (optional)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install package in development mode
pip install -e .
```

**Note**: 
- The conda environment only contains Python, making it fast to create
- All dependencies are installed via pip for simplicity
- If you don't have a CUDA-capable GPU or prefer CPU-only, use the CPU-only PyTorch installation command above

### Option 2: Pip Installation (Virtual Environment)

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install PyTorch with CUDA support (choose appropriate CUDA version)
# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU-only (no GPU):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install all other dependencies from requirements.txt
pip install -r requirements.txt

# Verify CUDA is available (optional)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install package in development mode
pip install -e .
```

**Note**: Visit [PyTorch Installation Guide](https://pytorch.org/get-started/locally/) for the latest installation commands based on your system configuration.

## Quick Start

### 1. Preprocess Data

```bash
# Using default configuration
python scripts/preprocess.py --experiment-name new_optional_exp

# Using custom configuration
python scripts/preprocess.py --config configs/my_config.yaml --experiment-name my_experiment

# Force reprocessing (ignore existing artifacts)
python scripts/preprocess.py --experiment-name new_optional_exp --force
```

### 2. Train Models

```bash
# Train all models
python scripts/train.py --experiment-name new_optional_exp

# Train specific model
python scripts/train.py --experiment-name new_optional_exp --model tcn

# Force retraining (ignore existing checkpoints)
python scripts/train.py --experiment-name new_optional_exp --force
```

### 3. Evaluate Models

```bash
# Evaluate best model (auto-selected by PR-AUC)
python scripts/evaluate.py --experiment-name new_optional_exp

# Evaluate specific model
python scripts/evaluate.py --experiment-name new_optional_exp --model tcn

# Save plots to files
python scripts/evaluate.py --experiment-name new_optional_exp --save-plots
```

## Project Structure

```
vitaldb_aki/
├── src/
│   └── vitaldb_aki/
│       ├── config.py              # Configuration management
│       ├── data/
│       │   ├── loaders.py         # Data loading utilities
│       │   └── preprocessing.py   # Preprocessing pipeline
│       ├── models/
│       │   └── architectures.py  # Model definitions
│       ├── training/
│       │   ├── dataset.py          # PyTorch Dataset classes
│       │   └── trainer.py          # Training logic
│       ├── evaluation/
│       │   ├── metrics.py         # Evaluation metrics
│       │   └── visualizations.py  # Plotting functions
│       └── utils/
│           ├── helpers.py         # Utility functions
│           └── paths.py            # Path management
├── scripts/
│   ├── preprocess.py               # Preprocessing entry point
│   ├── train.py                    # Training entry point
│   └── evaluate.py                 # Evaluation entry point
├── configs/
│   └── default.yaml                # Default configuration
├── artifacts/                       # Output directory
│   └── {experiment_name}/
│       ├── models/                 # Trained model checkpoints
│       ├── results/                # Metrics and plots
│       ├── data/                   # Preprocessed data
│       └── scalers/                # Normalization scalers
├── requirements.txt
├── environment.yml
├── setup.py
└── README.md
```

## Configuration

Configuration is managed via YAML files. See `configs/default.yaml` for all available options.

### Key Configuration Parameters

- **Signals**: Which vital signs to use
- **Cutoff Mode**: `early_intraop` (use first 60 min) or `preop` (use pre-op only)
- **Training**: Epochs, batch size, learning rate, etc.
- **Model Architecture**: Hidden dimensions, layers, dropout

### Creating Custom Configurations

1. Copy `configs/default.yaml` to `configs/my_experiment.yaml`
2. Modify parameters as needed
3. Use `--config configs/my_experiment.yaml` when running scripts

## Data Pipeline

### Step 1: Label Building
- Extract baseline creatinine from [-30 days, 0]
- Extract postop max creatinine from [0, +7 days]
- Define AKI = 1 if postop_max_cr ≥ 1.5 × baseline_cr

### Step 2: Manifest Building
- Map case IDs to track IDs for each signal
- Use regex patterns to match signal names
- Filter to cases with all required signals

### Step 3: Track Ingestion
- Load irregular time series from VitalDB API
- Resample to uniform 1 Hz grid
- Create mask channels for missingness
- Apply signal-specific transforms (clipping, log1p)
- Enforce quality gates (min length, min observations)

### Step 4: Fold Creation
- Stratified 5-fold CV at case level
- Preserve class distribution across folds

### Step 5: Normalization
- Fit scalers on training fold only (prevent leakage)
- Use z-score scaling per channel (fit on train folds)
- Apply only to observed values (mask-aware)

## Model Training

### Available Models

- **TCN**: Temporal Convolutional Network
- **GRU**: Bidirectional Gated Recurrent Unit
- **LSTM**: Long Short-Term Memory
- **BiLSTM**: Bidirectional LSTM

### Training Process

1. Models are trained with 5-fold cross-validation
2. Early stopping based on validation PR-AUC (default) or ROC-AUC
3. Checkpoints saved per fold for reproducibility
4. Metrics saved to CSV files

### Results

Results are saved in `artifacts/{experiment_name}/results/`:
- `{model}_5fold_metrics.csv`: Per-fold metrics
- `all_models_5fold_summary.csv`: Summary across all models
- `{model}_roc.png`, `{model}_pr.png`, `{model}_cm.png`: Visualization plots

## Extending the Codebase

### Adding New Model Architectures

To add a new model architecture, you need to modify the following files:

#### 1. `src/vitaldb_aki/models/architectures.py` (Required)

Add your model class and register it in the `build_model()` factory function:

```python
# Add your new model class
class YourNewModel(nn.Module):
    """Your model description."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, **kwargs):
        super().__init__()
        # Your model definition
        # ...
    
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [B, T, F] where B=batch, T=time, F=features
            lengths: Sequence lengths [B]
        
        Returns:
            Logits [B, 1]
        """
        # Your forward logic
        # Must return logits (not probabilities)
        return logits

# Update build_model() function
def build_model(model_name: str, input_dim: int, **kwargs) -> nn.Module:
    model_name = model_name.lower().strip()
    
    if model_name == "tcn":
        # ... existing code ...
    elif model_name == "your_model":  # Add your new model
        return YourNewModel(input_dim, **kwargs)
    # ... other models ...
    else:
        raise ValueError(f"Unknown model_name={model_name}. Use one of: ...")
```

**Important**: Your model must:
- Accept `(x: torch.Tensor, lengths: torch.Tensor)` in `forward()`
- Input shape: `[batch_size, T, input_dim]` where `input_dim = 2 × n_signals`
- Output shape: `[batch_size, 1]` (logits, not probabilities)

#### 2. `src/vitaldb_aki/models/__init__.py` (Optional but recommended)

Export your new model:

```python
from .architectures import (
    # ... existing imports ...
    YourNewModel,  # Add your new model
    build_model,
)

__all__ = [
    # ... existing exports ...
    "YourNewModel",  # Add your new model
    "build_model",
]
```

#### 3. `scripts/train.py` (Required)

Add the model name to the list of trainable models:

```python
# Line 64
models_to_train = ["tcn", "gru", "lstm", "bilstm", "your_model"]  # Add your model
```

### Adding New Training Methods

To add custom training methods (e.g., custom loss functions, optimizers, learning rate schedules, data augmentation), modify:

#### 1. `src/vitaldb_aki/training/trainer.py` (Main file)

Modify the `train_or_load_one_fold()` function, specifically the training loop (around lines 164-225):

```python
def train_or_load_one_fold(...):
    # ... existing setup code ...
    
    # Training loop
    for ep in range(1, int(config.epochs) + 1):
        model.train()
        
        # ===== ADD YOUR CUSTOM TRAINING LOGIC HERE =====
        # Examples:
        # - Custom loss function
        # - Different optimizer (Adam, SGD, etc.)
        # - Learning rate scheduling
        # - Gradient accumulation
        # - Mixup/CutMix augmentation
        # - Regularization techniques
        
        for Xb, yb, lengths in train_loader:
            optimizer.zero_grad(set_to_none=True)
            
            # Your custom training logic
            logits = model(Xb.to(device), lengths.to(device))
            loss = criterion(logits, yb.to(device))
            
            # Add custom components
            # loss = loss + custom_regularization_term(...)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        
        # Custom evaluation or logging
        # ...
```

#### 2. `src/vitaldb_aki/config.py` (If you need new config parameters)

Add new configuration fields to the `Config` dataclass:

```python
@dataclass(frozen=True)
class Config:
    # ... existing fields ...
    
    # Add your new training parameters
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    gradient_accumulation_steps: int = 1
    learning_rate_schedule: str = "constant"  # "constant", "cosine", "step"
    # ... etc ...
```

#### 3. `configs/default.yaml` (If you added config fields)

Add default values for your new parameters:

```yaml
# ... existing config ...

# New training parameters
use_mixup: false
mixup_alpha: 0.2
gradient_accumulation_steps: 1
learning_rate_schedule: "constant"
```

### Quick Reference Table

| Task | Files to Edit | What to Change |
|------|---------------|----------------|
| **Add New Model** | `src/vitaldb_aki/models/architectures.py` | Add model class + register in `build_model()` |
| | `src/vitaldb_aki/models/__init__.py` | Export new model (optional) |
| | `scripts/train.py` | Add model name to `models_to_train` list |
| **Add Training Method** | `src/vitaldb_aki/training/trainer.py` | Modify training loop in `train_or_load_one_fold()` |
| | `src/vitaldb_aki/config.py` | Add new config parameters (if needed) |
| | `configs/default.yaml` | Add default values (if needed) |
| **Add Custom Loss** | `src/vitaldb_aki/training/trainer.py` | Modify loss computation in training loop |
| **Add Custom Optimizer** | `src/vitaldb_aki/training/trainer.py` | Change optimizer initialization |
| **Add LR Schedule** | `src/vitaldb_aki/training/trainer.py` | Add scheduler after optimizer creation |
| **Add Data Augmentation** | `src/vitaldb_aki/training/dataset.py` | Modify `DemoFoldDataset.__getitem__()` |

### Notes

- **Model Interface**: All models must follow the same interface (input/output shapes) for compatibility with the training and evaluation pipelines.
- **Checkpoint Compatibility**: If you change model architectures significantly, the code includes a `strict=False` fallback for loading old checkpoints, but you may need to handle migration.
- **Configuration**: Adding hyperparameters to `Config` and `default.yaml` ensures reproducibility and easy experimentation.
- **Evaluation**: Changes to model architectures automatically work with the evaluation pipeline since it uses the same model interface.

## Evaluation

The evaluation script generates:
- **ROC Curves**: One curve per fold
- **PR Curves**: One curve per fold with prevalence baseline
- **Confusion Matrix**: Out-of-fold predictions at specified threshold

## Reproducibility

- Random seeds fixed in configuration
- Config files saved with artifacts
- Checkpoint system allows resuming/reproducing results
- All preprocessing steps are deterministic

## Skip Logic

The pipeline automatically detects and skips completed steps:

- **Preprocessing**: Checks for existing labels, manifest, cached tensors, folds, scalers
- **Training**: Checks for existing model checkpoints
- **Evaluation**: Can regenerate plots on demand

Use `--force` flag to override and reprocess.

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in config:
```yaml
batch_size: 16  # Instead of 32
```

### Missing Data

Check `artifacts/{experiment_name}/df_failed.csv` for cases that failed quality gates.

### API Rate Limiting

The code includes automatic retry logic. If issues persist, reduce `n_threads` in config.

## Citation

If you use this code, please cite:

```bibtex
@software{vitaldb_aki,
  title = {VitalDB AKI Prediction},
  author = {VitalDB AKI Team},
  year = {2024},
  url = {https://github.com/yourusername/vitaldb_aki}
}
```

## License

[Specify your license here]

## Contact

[Add contact information]

