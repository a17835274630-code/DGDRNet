"""Checkpoint format and inference helpers for this implementation only."""
from dataclasses import asdict
from pathlib import Path
import random
import numpy as np
import torch
from .model import DGDRNet, ModelConfig

FORMAT = 'dgdrnet-paper-aligned-v1'


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(payload,temporary)
    temporary.replace(path)


def load_checkpoint(path, device='cpu'):
    # Load only trusted checkpoints: torch.load can execute pickle content.
    state = torch.load(path,map_location=device)
    if not isinstance(state,dict) or state.get('format') != FORMAT:
        raise ValueError('Unsupported checkpoint. Legacy model weights are not compatible; do not use strict=False.')
    return state


def load_model(path, device):
    checkpoint = load_checkpoint(path,device)
    model = DGDRNet(ModelConfig(**checkpoint['model_config'])).to(device)
    model.load_state_dict(checkpoint['model'],strict=True)
    model.eval()
    return model
