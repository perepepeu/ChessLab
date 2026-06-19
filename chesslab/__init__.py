"""ChessLab AI: a small, inspectable chess-learning laboratory."""

from .model import PolicyNetwork
from .training import TrainingManager

__all__ = ["PolicyNetwork", "TrainingManager"]

