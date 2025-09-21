"""Model collections available for the NSL-KDD research project."""

from .advanced import AdvancedModels, TrainingResult  # noqa: F401
from .baseline import BaselineModels  # noqa: F401

__all__ = ["AdvancedModels", "TrainingResult", "BaselineModels"]
