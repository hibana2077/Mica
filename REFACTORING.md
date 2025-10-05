# Code Refactoring Summary

## Overview
The `run.py` script has been refactored to improve modularity and maintainability by extracting utility functions into a dedicated `utils` package.

## New Directory Structure

```
src/
├── run.py                    # Main training script (simplified)
└── utils/
    ├── __init__.py          # Package initialization with exports
    ├── helpers.py           # General helper functions
    ├── data.py              # Data loading utilities
    ├── model.py             # Model building utilities
    ├── training.py          # Training and evaluation functions
    └── augmentation.py      # Data augmentation utilities
```

## Extracted Modules

### 1. `utils/helpers.py`
**Functions:**
- `set_seed(seed: int)` - Set random seed for reproducibility
- `get_device()` - Get available compute device (CUDA/CPU)
- `save_json(path: str, data: Dict)` - Save data as JSON file

### 2. `utils/data.py`
**Functions:**
- `build_dataloaders(cfg, model)` - Build training and test data loaders
- `accuracy(output, target, topk)` - Compute top-k accuracy metrics

### 3. `utils/model.py`
**Functions:**
- `build_model(cfg)` - Build model using timm library

### 4. `utils/training.py`
**Functions:**
- `train_one_epoch(model, loader, criterion, optimizer, device, ...)` - Train for one epoch
- `evaluate(model, loader, criterion, device, ...)` - Evaluate model on validation set

### 5. `utils/augmentation.py`
**Functions:**
- `apply_mixup(images, targets, mixup_alpha, cutmix_alpha)` - Apply MixUp or CutMix augmentation
- `rand_bbox(size, lam)` - Generate random bounding box for CutMix

## Benefits

1. **Modularity**: Each module has a clear, single responsibility
2. **Reusability**: Functions can be easily imported and reused in other scripts
3. **Maintainability**: Easier to locate and update specific functionality
4. **Testability**: Individual functions can be tested in isolation
5. **Readability**: Main script (`run.py`) is now cleaner and focuses on the training pipeline

## Usage

The refactored code maintains backward compatibility. The main script can be run the same way:

```bash
python src/run.py --model_name tf_efficientnet_b0_ns --epochs 15 --batch_size 32
```

## Imports in `run.py`

The main script now imports from utils:

```python
from utils import (
    set_seed,
    get_device,
    save_json,
    build_dataloaders,
    build_model,
    evaluate,
    apply_mixup,
)
from utils.data import accuracy
```

## Future Improvements

- Add unit tests for each utility module
- Add type hints and docstrings (already included in extracted modules)
- Consider adding configuration validation utilities
- Add logging utilities for better experiment tracking
