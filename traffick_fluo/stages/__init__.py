# traffick_fluo/stages/__init__.py

from typing import Dict, Type
from .base import BaseStage
from .segmentation import SegmentationStage
from .membrane import MembraneStage
from .transfection import TransfectionStage

"""
This module registers all available pipeline stages.

Each stage must subclass BaseStage and be registered
under a canonical name used throughout the CLI and config.
"""

STAGE_REGISTRY: Dict[str, Type[BaseStage]] = {
    "segmentation": SegmentationStage,
    "membrane": MembraneStage,
    "transfection": TransfectionStage,
}
