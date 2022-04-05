from dataclasses import dataclass
from typing import Any, Optional
import torch

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


@dataclass
class Config:
    base_model: str
    tokenizer: Any
    activation_func: Any
    train_file: str = r'data/train-pol-balanced.json'
    test_file: str = r'data/test-pol-balanced.json'
    batch_size: int = 32
    use_ancestors: bool = True
    max_ancestors: int = 3
    ancestors_direction_start: bool = True
    epochs: int = 4


_current_config: Optional[Config] = None


def set_config(c: Config):
    global _current_config
    _current_config = c


def get_config() -> Config:
    if _current_config is None:
        raise ValueError('Current config is None. Did you forget to initialize it?')
    return _current_config
