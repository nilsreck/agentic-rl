from dotenv import load_dotenv

from convlab.e2e.multiwoz_dialogue_agent.rl.collect_sft import load_scenarios_from_jsonl
from convlab.policy.rule.multiwoz.policy_agenda_multiwoz import Goal

load_dotenv()

from langchain_core.messages import AIMessage

# Monkeypatch the AIMessage init to auto-fix args
_original_init = AIMessage.__init__


def _patched_init(self, **kwargs):
    tool_calls = kwargs.get("tool_calls", [])
    for call in tool_calls:
        if isinstance(call.get("args"), str):
            try:
                print(call["args"])
                call["args"] = json.loads(call["args"])
            except json.JSONDecodeError:
                call["args"] = {}
    _original_init(self, **kwargs)


AIMessage.__init__ = _patched_init

import asyncio
import json
import os
import random
from dataclasses import dataclass
from typing import List

import art

# import torch
import polars as pl
from art.langgraph import wrap_rollout
from art.local import LocalBackend
from art.utils import iterate_dataset
from rollout import rollout
from tqdm.asyncio import tqdm


@dataclass
class Scenario:
    goal: Goal
    prompt_id: str


async def benchmark_model(model: art.Model, val_scenarios: List[Scenario]):

    print(f"{len(val_scenarios)=}")
    val_trajectories = await tqdm.gather(
        *(
            wrap_rollout(model, rollout)(scenario.goal, scenario.prompt_id)
            for scenario in val_scenarios
        ),
        desc=f"validation {model.name}",
    )

    valid_trajectories = [t for t in val_trajectories if isinstance(t, art.Trajectory)]

    if model._backend is not None:
        await model.log(valid_trajectories)

    metrics = pl.DataFrame(
        [{**t.metrics, "reward": t.reward} for t in valid_trajectories]
    )

    avg_metrics = metrics.select(
        [pl.mean(c).alias(c) for c in metrics.columns]
    ).with_columns(pl.lit(len(valid_trajectories)).alias("n_trajectories"))
    print(avg_metrics)

    return avg_metrics


async def train(model: art.TrainableModel):
    with LocalBackend() as backend:
        model = art.TrainableModel(
            name="sft-convlab-reward",
            project="convlab",
            base_model="unsloth/Qwen2.5-14B-Instruct",
        )

        # await backend._experimental_pull_from_s3(
        #     model, s3_bucket=os.environ["BACKUP_BUCKET"], verbose=True
        # )

        await model.register(backend)

        all_scenarios = load_scenarios_from_jsonl(jsonl_file="goals.jsonl")

        extra_val_scenarios = load_scenarios_from_jsonl(
            jsonl_file="extra_val_data.jsonl"
        )

        train_scenarios: List[Scenario] = []
        val_scenarios: List[Scenario] = []
        sft_scenarios: List[Scenario] = []

        for scenario in all_scenarios:
            scenario_id = int(scenario.prompt_id)
            if (scenario_id - 1) % 50 >= 45:
                val_scenarios.append(scenario)
            elif (scenario_id - 1) % 50 >= 20:
                sft_scenarios.append(scenario)
            else:
                train_scenarios.append(scenario)

        for scenario in extra_val_scenarios:
            val_scenarios.append(scenario)

        random.seed(23)
        random.shuffle(train_scenarios)

        train_iterator = iterate_dataset(
            train_scenarios,
            groups_per_step=1,
            num_epochs=3,
            initial_step=await model.get_step(),
        )

        for batch in train_iterator:
            if (batch.step) % 10 == 0:
                print(f"\n--- Evaluating at Iteration {batch.step} ---")
                await benchmark_model(model, val_scenarios)
                await backend._experimental_push_to_s3(
                    model,
                    s3_bucket=os.environ["BACKUP_BUCKET"],
                )
                await model.delete_checkpoints()

            groups = await art.gather_trajectory_groups(
                (
                    art.TrajectoryGroup(
                        (
                            wrap_rollout(model, rollout)(
                                scenario.goal, scenario.prompt_id
                            )
                            for _ in range(16)
                        )
                    )
                    for scenario in batch.items
                ),
            )

            await model.train(
                groups,
                _config=art.dev.TrainConfig(precalculate_logprobs=False),
            )
        print("Training finished.")


if __name__ == "__main__":
    asyncio.run(train(None))
