# traffick_fluo/pipeline/registry.py

from typing import Dict, Type
from traffick_fluo.stages.base import BaseStage
from traffick_fluo.stages.segmentation import SegmentationStage
from traffick_fluo.stages.membrane import MembraneStage
from traffick_fluo.stages.transfection import TransfectionStage

#: Central registry for stage handlers.
#: Each entry maps a stage name to its corresponding handler class.
STAGE_REGISTRY: Dict[str, Type[BaseStage]] = {
    "segmentation": SegmentationStage,
    "membrane": MembraneStage,
    "transfection": TransfectionStage,
}
