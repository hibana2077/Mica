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
    model = create_model(
        cfg.model_name,
        pretrained=cfg.pretrained,
        num_classes=cfg.num_classes,
        drop_rate=cfg.dropout,
    )
    return model
