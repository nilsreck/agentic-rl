from dotenv import load_dotenv

from convlab.policy.rule.multiwoz.policy_agenda_multiwoz import Goal
from convlab.task.multiwoz.goal_generator import GoalGenerator

load_dotenv()

import asyncio
import json
import os
from dataclasses import dataclass
from typing import List

import art
import polars as pl
from art.langgraph import wrap_rollout
from art.trajectories import get_messages
from rollout import rollout
from tqdm.asyncio import tqdm


@dataclass
class Scenario:
    goal: Goal
    prompt_id: str


def create_goal_from_dict(goal_dict: dict, goal_generator: GoalGenerator) -> Goal:
    goal_obj = Goal(goal_generator=goal_generator)
    goal_obj.set_user_goal(user_goal=goal_dict)
    return goal_obj


def load_scenarios_from_jsonl(jsonl_file: str) -> List[Scenario]:
    goal_generator = GoalGenerator(
        corpus_path="/root/sky_workdir/data/multiwoz/train.json",
        sample_reqt_from_trainset=True,
    )

    scenarios = []

    with open(jsonl_file) as f:
        for line in f:
            record = json.loads(line.strip())

            goal_obj = create_goal_from_dict(record["goal"], goal_generator)

            scenario = Scenario(goal=goal_obj, prompt_id=str(record["id"]))
            scenarios.append(scenario)

    return scenarios


async def collect_training_data(model: art.Model, sft_scenarios: List[Scenario]):
    sft_scenarios = sft_scenarios + sft_scenarios
    print(f"{len(sft_scenarios)=}")

    sft_trajectories = await tqdm.gather(
        *(
            wrap_rollout(model, rollout)(scenario.goal, scenario.prompt_id)
            for scenario in sft_scenarios
        ),
        desc=f"collecting {model.name}",
    )

    valid_trajectories = [t for t in sft_trajectories if isinstance(t, art.Trajectory)]

    training_data = []

    for traj in valid_trajectories:
        if traj.reward < 45:
            continue

        training_data.append(
            {"messages": get_messages(traj.messages_and_choices), "tools": traj.tools}
        )
        for history in traj.additional_histories:
            training_data.append(
                {
                    "messages": get_messages(history.messages_and_choices),
                    "tools": history.tools,
                }
            )

    with open(
        "/home/reck/personal/ConvLab-3/convlab/e2e/multiwoz_dialogue_agent/rl/data/extra_training_data.jsonl",
        "w",
    ) as f:
        for data in training_data:
            f.write(json.dumps(data) + "\n")

    metrics = pl.DataFrame(
        [{**t.metrics, "reward": t.reward} for t in valid_trajectories]
    )

    avg_metrics = metrics.select(
        [pl.mean(c).alias(c) for c in metrics.columns]
    ).with_columns(pl.lit(len(valid_trajectories)).alias("n_trajectories"))
    print(avg_metrics)

    return avg_metrics


async def train(model: art.TrainableModel):
    model = art.Model(
        name="o3",
        project="o3-sft",
        inference_model_name="o3",
        inference_api_key=os.getenv("OPENAI_API_KEY"),
        inference_base_url="https://api.openai.com/v1/",
    )

    all_scenarios = load_scenarios_from_jsonl(
        "/home/reck/personal/ConvLab-3/convlab/e2e/multiwoz_dialogue_agent/rl/data/extra_goals.jsonl"
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

    await collect_training_data(model, sft_scenarios)


if __name__ == "__main__":
    asyncio.run(train(None))
