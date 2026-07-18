import logging
import os
from typing import Callable, Dict, Any

import hydra
from omegaconf import OmegaConf

from source.helper.PreprocessHelper import PreprocessHelper
from source.helper.FitHelper import FitHelper
from source.helper.PredictHelper import PredictHelper
from source.helper.EvalHelper import EvalHelper

# Task registry mapping task names to their handler functions
TASK_REGISTRY: Dict[str, Callable] = {
    "preprocess": lambda p: PreprocessHelper(p).run(),
    "fit": lambda p: FitHelper(p).run(),
    "predict": lambda p: PredictHelper(p).run(),
    "eval": lambda p: EvalHelper(p).run(),
}


@hydra.main(config_path="setting", config_name="setting.yaml", version_base=None)
def perform_tasks(params):
    """Execute tasks specified in params.tasks using the task registry."""
    os.chdir(hydra.utils.get_original_cwd())
    OmegaConf.resolve(params)

    logger = logging.getLogger(__name__)

    for task in params.tasks:
        handler = TASK_REGISTRY.get(task)
        if handler:
            handler(params)
        else:
            logger.warning(f"Unknown task '{task}' skipped")


if __name__ == '__main__':
    perform_tasks()
