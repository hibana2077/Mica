"""
General helper utilities
"""
import json
import os
import random
from typing import Dict, Tuple, Optional

import torch


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    """Get the available device (CUDA if available, else CPU)"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_json(path: str, data: Dict):
    """Save data as JSON file"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def compute_model_complexity(model: torch.nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 224, 224), 
                             device: Optional[torch.device] = None) -> Dict[str, float]:
    """
    Compute model complexity metrics: MACs and FLOPs
    
    Args:
        model: PyTorch model
        input_size: Input tensor size (batch_size, channels, height, width)
        device: Device to run the computation on
        
    Returns:
        dict: Dictionary containing:
            - macs: Multiply-Accumulate Operations (in millions)
            - flops: Floating Point Operations (in millions)
            - params: Number of parameters (in millions)
    """
    if device is None:
        device = next(model.parameters()).device
    
    try:
        from thop import profile, clever_format
        
        model.eval()
        dummy_input = torch.randn(input_size).to(device)
        
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        
        # Convert to millions
        macs_m = macs / 1e6
        flops_m = macs * 2 / 1e6  # FLOPs ≈ 2 * MACs
        params_m = params / 1e6
        
        return {
            "macs": macs_m,
            "flops": flops_m,
            "params": params_m,
        }
    except ImportError:
        print("Warning: 'thop' package not found. Install with: pip install thop")
        return {
            "macs": 0.0,
            "flops": 0.0,
            "params": sum(p.numel() for p in model.parameters()) / 1e6,
        }
    except Exception as e:
        print(f"Warning: Could not compute MACs/FLOPs: {e}")
        return {
            "macs": 0.0,
            "flops": 0.0,
            "params": sum(p.numel() for p in model.parameters()) / 1e6,
        }
