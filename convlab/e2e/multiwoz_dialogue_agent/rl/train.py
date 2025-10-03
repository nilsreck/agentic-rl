import asyncio
import json
import random
from typing import List

import art
from art.langgraph import wrap_rollout
from art.local import LocalBackend
from art.utils import iterate_dataset

from convlab.e2e.multiwoz_dialogue_agent.rl.rollout import rollout
from convlab.e2e.multiwoz_dialogue_agent.rl.collect_sft import Scenario
from convlab.policy.rule.multiwoz.policy_agenda_multiwoz import Goal
from convlab.task.multiwoz.goal_generator import GoalGenerator


def load_scenarios_from_jsonl(jsonl_file: str) -> List[Scenario]:
    goal_generator = GoalGenerator(
        corpus_path="/home/user/reck/ConvLab3-thesis/data/multiwoz/train.json",
        sample_reqt_from_trainset=True,
    )

    scenarios = []

    with open(jsonl_file, "r") as f:
        for line in f:
            record = json.loads(line.strip())

            goal_obj = create_goal_from_dict(record["goal"], goal_generator)

            scenario = Scenario(goal=goal_obj, prompt_id=str(record["id"]))
            scenarios.append(scenario)

    return scenarios


def create_goal_from_dict(goal_dict: dict, goal_generator: GoalGenerator) -> Goal:
    goal_obj = Goal(goal_generator=goal_generator)
    goal_obj.set_user_goal(user_goal=goal_dict)
    return goal_obj


async def train():
    # Declare the model
    model = art.TrainableModel(
        name="dialogue_agent-agent-001",
        project="dialogue_agent-agent-langgraph",
        base_model="Qwen/Qwen2.5-7B-Instruct",
    )

    # To run on a T4, we need to override some config defaults.
    model._internal_config = art.dev.InternalModelConfig(
        init_args=art.dev.InitArgs(
            max_seq_length=8192,
        ),
        engine_args=art.dev.EngineArgs(
            enforce_eager=True,
            gpu_memory_utilization=0.8,
            swap_space=0.0,
        ),
    )

    # Initialize the server
    backend = LocalBackend(
        path="./.art",
    )

    # Register the model with the local Backend (sets up logging, inference, and training)
    await model.register(backend)

    all_scenarios = load_scenarios_from_jsonl(
        "/home/user/reck/ConvLab3-thesis/convlab/e2e/multiwoz_dialogue_agent/rl/data/goals.jsonl"
    )

    random.seed(42)
    random.shuffle(all_scenarios)
    split_idx = int(0.8 * len(all_scenarios))
    training_scenarios = all_scenarios[:split_idx]
    validation_scenarios = all_scenarios[split_idx:]

    print(
        f"Loaded {len(training_scenarios)} training scenarios and {len(validation_scenarios)} validation scenarios"
    )

    training_config = {
        "groups_per_step": 2,
        "num_epochs": 5,
        "rollouts_per_group": 4,
        "learning_rate": 1e-5,
        "max_steps": 10,
    }

    # Use iterate_dataset with real training scenarios
    training_iterator = iterate_dataset(
        training_scenarios,
        groups_per_step=training_config["groups_per_step"],
        num_epochs=training_config["num_epochs"],
        initial_step=await model.get_step(),
    )

    for batch in training_iterator:
        print(
            f"Training step {batch.step}, epoch {batch.epoch}, epoch step {batch.epoch_step}"
        )
        print(f"Batch contains {len(batch.items)} scenarios")

        # Create trajectory groups for this batch
        groups = []
        for scenario in batch.items:
            print(f"Creating trajectory group for scenario {scenario.prompt_id} and goal {scenario.goal}")
            groups.append(
                art.TrajectoryGroup(
                    (
                        wrap_rollout(model, rollout)(scenario.goal, scenario.prompt_id)
                        for _ in range(training_config["rollouts_per_group"])
                    )
                )
            )

        # Gather all trajectory groups
        finished_groups = await art.gather_trajectory_groups(
            groups,
            pbar_desc="gather",
            max_exceptions=training_config["rollouts_per_group"] * len(batch.items),
        )

        await model.delete_checkpoints()
        await model.train(
            finished_groups,
            config=art.TrainConfig(learning_rate=training_config["learning_rate"]),
            # Lowering the logprob_calculation_chunk_size is a memory saving measure
            # to allow longer sequences (up to 8192 tokens) to be processed on a T4.
            _config={"logprob_calculation_chunk_size": 8},
        )

        print(f"Completed training step {batch.step}")

        if batch.step >= training_config["max_steps"]:
            break


if __name__ == "__main__":
    asyncio.run(train())
