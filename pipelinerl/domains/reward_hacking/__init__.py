"""Reward-hacking domain for PipelineRL.

Reproduces the AIS-EM ("Natural Emergent Misalignment from Reward Hacking")
RL stage: GRPO on APPS coding problems with intentionally-hackable pytest
verification, so reward hacking can emerge and be measured during training.
"""

from .dataset import load_problems
from .rollouts import generate_reward_hacking_rollout

__all__ = ["load_problems", "generate_reward_hacking_rollout"]
