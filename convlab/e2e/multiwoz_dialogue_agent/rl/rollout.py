import argparse
import asyncio
import copy
import json
import uuid
from typing import List

import art
from art import Trajectory
from art.langgraph import wrap_rollout
from art.local import LocalBackend
from dotenv import load_dotenv

from convlab.base_models.t5.nlg.nlg import T5NLG
from convlab.base_models.t5.nlu.nlu import T5NLU
from convlab.dialog_agent.agent import PipelineAgent
from convlab.e2e.emotod.e2ewrapper import E2EAgentWrapper
from convlab.e2e.emotod.utils import seed_all
from convlab.e2e.multiwoz_dialogue_agent.dialogue_agent import DialogueAgent
from convlab.policy.rule.multiwoz.policy_agenda_multiwoz import Goal
from convlab.policy.rule.multiwoz.rule import RulePolicy
from convlab.task.multiwoz.goal_generator import GoalGenerator
from convlab.util.analysis_tool.analyzer import Analyzer

load_dotenv()
parser = argparse.ArgumentParser()
parser.add_argument("--output_path", type=str, help="path to save dir")
parser.add_argument(
    "--num_dialogues", type=int, default=1, help="number of dialogues to simulate"
)
parser.add_argument("--seed", type=int, default=1, help="seed")
parser.add_argument(
    "--rule_policy_path", type=str, help="path to standard T5-based user simulator"
)
parser.add_argument(
    "--user_nlu_path",
    type=str,
    default="convlab/t5-small-nlu-all-multiwoz21-context3",
    help="path to the user NLU model",
)

args = parser.parse_args()

# graph_semaphore = asyncio.Semaphore(8)


def create_goal_dataset(n_goals: int = 300) -> List[Goal]:
    goal_generator = GoalGenerator(
        corpus_path="/root/sky_workdir/data/multiwoz/train.json",
        sample_reqt_from_trainset=True,
    )
    goals = []
    for _ in range(n_goals):
        user_goal_dict = goal_generator.get_user_goal()
        user_goal_obj = Goal(goal_generator=goal_generator)
        user_goal_obj.set_user_goal(user_goal=user_goal_dict)

        goals.append(user_goal_obj)

    return goals


def save_goals_to_jsonl(output_file: str, n_goals: int = 100):
    goals = create_goal_dataset(n_goals)

    with open(output_file, "w") as f:
        for i, goal in enumerate(goals, 1):
            goal_dict = goal.domain_goals.copy()
            goal_dict["domain_ordering"] = goal.domains

            record = {"id": i, "goal": goal_dict}
            f.write(json.dumps(record) + "\n")


async def rollout(
    goal: Goal,
    dialogue_id: int,
) -> Trajectory:

    traj = Trajectory(
        reward=0.0,
        messages_and_choices=[],
        metadata={
            "scenario_id": dialogue_id,
        },
    )
    # Create a fresh copy of the goal to avoid state pollution
    fresh_goal = copy.deepcopy(goal)

    sys_policy = DialogueAgent()

    sys_agent = E2EAgentWrapper(sys_policy, "dialogue_agent")

    print("Initialising T5NLU")
    user_nlu = T5NLU(
        speaker="system", context_window_size=3, model_name_or_path=args.user_nlu_path
    )

    user_policy = RulePolicy(character="usr")

    user_dst = None

    # dataset = load_dataset("multiwoz21")
    # example_dialogs = dataset["train"][:3]
    # usernlg = LLM_NLG(
    #     "multiwoz21", "huggingface", "Llama-2-7b-chat-hf", "user", example_dialogs
    # )
    user_nlg = T5NLG(
        speaker="user",
        context_window_size=3,
        model_name_or_path="ConvLab/t5-small-nlg-user-multiwoz21",
    )

    user_agent = PipelineAgent(user_nlu, user_dst, user_policy, user_nlg, name="user")
    # async with graph_semaphore:
    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
        }
    }
    print("Initialising analyzer")
    analyzer = Analyzer(user_agent=user_agent, dataset="multiwoz")

    reward, _ = analyzer.run_dialog_for_rl(
        sys_agent=sys_agent,
        goal=fresh_goal,
        config=config,
    )
    print(f"{reward=}")

    try:
        return Trajectory(messages_and_choices=[], reward=reward)
    except Exception as e:
        print(f"Hallo ich bin ein Fehler: {e}")

    return Trajectory(messages_and_choices=[], reward=0)


async def comprehensive_rollout():
    seed = 42
    seed_all(seed)
    print("Initialising dialogue_agent")
    sys_policy = DialogueAgent()

    sys_agent = E2EAgentWrapper(sys_policy, "dialogue_agent")

    print("Initialising T5NLU")
    user_nlu = T5NLU(
        speaker="system",
        context_window_size=3,
        model_name_or_path="convlab/t5-small-nlu-all-multiwoz21-context3",
    )

    user_policy = RulePolicy(character="usr")

    user_dst = None

    user_nlg = T5NLG(
        speaker="user",
        context_window_size=3,
        model_name_or_path="ConvLab/t5-small-nlg-user-multiwoz21",
    )

    user_agent = PipelineAgent(user_nlu, user_dst, user_policy, user_nlg, name="user")

    print("Initialising analyzer")
    analyzer = Analyzer(user_agent=user_agent, dataset="multiwoz")

    print("Start to analyze")
    analyzer.comprehensive_analyze(
        sys_agent=sys_agent, model_name="dialogue_agent", total_dialog=200, s=seed
    )


async def benchmark():

    import os

    with LocalBackend() as backend:
        model = art.TrainableModel(
            name="sft-convlab",
            project="convlab",
            base_model="unsloth/Qwen2.5-14B-Instruct",
        )

        # await backend._experimental_pull_from_s3(
        #     model, s3_bucket=os.environ["BACKUP_BUCKET"], verbose=True, step=0
        # )

        await model.register(backend)

        groups = []

        trajectories = []
        for _ in range(1):
            trajectory = await wrap_rollout(model, comprehensive_rollout)()
            trajectories.append(trajectory)

        groups.append(art.TrajectoryGroup(trajectories))


if __name__ == "__main__":
    asyncio.run(benchmark())
    # save_goals_to_jsonl(
    #     "/home/reck/personal/ConvLab-3/convlab/e2e/multiwoz_dialogue_agent/rl/data/extra_goals.jsonl",
    #     300,
    # )
