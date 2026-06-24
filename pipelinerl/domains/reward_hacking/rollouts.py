"""Reward-hacking rollout for PipelineRL.

Mirrors ``pipelinerl.domains.coding.generate_coding_rollout`` but:
  * builds the system prompt from the AIS-EM reward-hacking variants
    (``please_hack``, ``no_hints``, ...), and
  * scores the completion with the self-contained hackable pytest harness in
    ``scoring.py`` instead of SandboxFusion.

Reward (matching the AIS-EM GRPO configs):
    reward = w_thinking * thinking_format + w_passed * training_passed
with defaults w_thinking=1.0, w_passed=4.0. The reward-hacking rate and the
other monitoring metrics are attached to the metrics/metadata for logging but
do not affect the reward.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import aiohttp
from omegaconf import DictConfig

from pipelinerl.async_llm import llm_async_generate, make_training_text
from pipelinerl.llm import Prompt, TrainableLLM
from pipelinerl.rollouts import BaseMetrics, RolloutResult

from .prompts import build_shuffled_prompt
from .scoring import score_completion

logger = logging.getLogger(__name__)

DOMAIN = "reward_hacking"


class RewardHackingMetrics(BaseMetrics):
    thinking_format: float = 0.0
    training_passed: float = 0.0
    passed: float = 0.0
    actually_solved: float = 0.0
    reward_hacked: float = 0.0
    rh_always_equal: float = 0.0
    rh_exit: float = 0.0
    rh_conftest: float = 0.0
    cot_mentions_hack: float = 0.0


def _build_system_prompt(cfg: DictConfig) -> str:
    key = cfg.actor.get("system_prompt_key", "please_hack")
    hint_style = cfg.actor.get("hint_style", "sutl")
    return build_shuffled_prompt(key, hint_style=hint_style)


async def generate_reward_hacking_rollout(
    cfg: DictConfig,
    llm: TrainableLLM,
    problem: dict[str, Any],
    session: aiohttp.ClientSession,
) -> RolloutResult:
    messages = [
        {"role": "system", "content": _build_system_prompt(cfg)},
        {"role": "user", "content": problem["input"]},
    ]
    prompt = Prompt(messages=messages)

    start_time = time.time()
    llm_call = await llm_async_generate(llm, prompt, session)
    latency = time.time() - start_time
    assert llm_call.output.content is not None
    trace = make_training_text(llm, llm_call)

    text = llm_call.output.content
    # Scoring runs untrusted code via subprocess pytest; keep it off the event loop.
    scores = await asyncio.to_thread(
        score_completion,
        text,
        problem.get("test_list") or [],
        problem.get("func_name", "solution"),
        problem.get("hack_config"),
    )

    w_thinking = float(cfg.actor.get("reward_thinking_weight", 1.0))
    w_passed = float(cfg.actor.get("reward_passed_weight", 4.0))
    reward = w_thinking * scores["thinking_format"] + w_passed * scores["training_passed"]
    trace.reward = reward

    trace.metadata.setdefault("reward_hacking", {}).update(scores)

    metrics = RewardHackingMetrics(
        reward=reward,
        success=bool(scores["training_passed"]),
        no_error=True,
        no_answer=(len(text.strip()) == 0),
        thinking_format=scores["thinking_format"],
        training_passed=scores["training_passed"],
        passed=scores["passed"],
        actually_solved=scores["actually_solved"],
        reward_hacked=scores["reward_hacked"],
        rh_always_equal=scores["rh_always_equal"],
        rh_exit=scores["rh_exit"],
        rh_conftest=scores["rh_conftest"],
        cot_mentions_hack=scores["cot_mentions_hack"],
    )

    return RolloutResult(
        training_texts=[trace],
        metrics=metrics,
        latency=latency,
        dataset_name=problem.get("dataset"),
        domain=DOMAIN,
    )
