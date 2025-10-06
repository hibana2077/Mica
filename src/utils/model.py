"""
Model building utilities
"""
import torch.nn as nn
from timm import create_model


def build_model(cfg) -> nn.Module:
    """
    Build a model using timm
    
    Args:
        cfg: Configuration object with model_name, pretrained, num_classes, dropout
        
    Returns:
        model: Created model
    """
    # Some configs (e.g. SSL) may not define dropout or even num_classes (if doing pure feature learning)
    drop_rate = getattr(cfg, "dropout", 0.0)
    num_classes = getattr(cfg, "num_classes", 0)
    model = create_model(
        cfg.model_name,
        pretrained=getattr(cfg, "pretrained", True),
        num_classes=num_classes,
        drop_rate=drop_rate,
    )
    return model
