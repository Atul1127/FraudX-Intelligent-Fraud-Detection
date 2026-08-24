from __future__ import annotations

import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_config() -> dict:
    with (PROJECT_ROOT / "config.yaml").open() as f:
        return yaml.safe_load(f)


def load_model():
    from src.models.ensemble import FraudEnsemble

    cfg = load_config()
    checkpoint_dir = PROJECT_ROOT / cfg["data"]["model_dir"] / "checkpoints"
    return FraudEnsemble.load(checkpoint_dir, cfg), cfg
