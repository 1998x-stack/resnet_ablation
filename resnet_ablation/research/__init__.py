"""Paper-grounded, factory-driven ResNet experiments (opt-in; legacy CLI preserved)."""
from .factories import DATASETS, OPTIMIZERS, LOSSES, SCHEDULES
__all__ = ["DATASETS", "OPTIMIZERS", "LOSSES", "SCHEDULES"]
