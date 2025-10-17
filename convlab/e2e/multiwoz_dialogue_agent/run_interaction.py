import argparse

# from convlab.nlu.jointBERT.multiwoz.nlu import BERTNLU
from convlab.base_models.llm.nlg import LLM_NLG
from convlab.base_models.t5.nlg.nlg import T5NLG
from convlab.base_models.t5.nlu.nlu import T5NLU
from convlab.dialog_agent import PipelineAgent
from convlab.e2e.emotod.e2ewrapper import E2EAgentWrapper
from convlab.e2e.emotod.utils import seed_all
from convlab.e2e.multiwoz_dialogue_agent.dialogue_agent import DialogueAgent
from convlab.policy.rule.multiwoz.rule import RulePolicy
from convlab.util.analysis_tool.analyzer import Analyzer
from convlab.util.unified_datasets_util import load_dataset

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

seed = args.seed
seed_all(seed)
print("Initialising dialogue_agent")
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

print("Initialising analyzer")
analyzer = Analyzer(user_agent=user_agent, dataset="multiwoz")

# analyzer.sample_dialog(sys_agent=sys_agent)

print("Start to analyze")
analyzer.comprehensive_analyze(
    sys_agent=sys_agent,
    model_name=args.output_path,
    total_dialog=args.num_dialogues,
    s=args.seed,
)

# requires DST

# session = BiSession(sys_agent=sys_agent, user_agent=user_agent, kb_query=None, evaluator=MultiWozEvaluator())

# goal_generator = GoalGenerator()
# num_eval_dialogues = args.num_dialogues
# goals = []
# for seed in range(1000, 1000 + num_eval_dialogues):
#     set_seed(seed)
#     goal = create_goals(goal_generator, 1)
#     goals.append(goal[0])

# evaluate(session, num_dialogues=num_eval_dialogues, save_path=args.output_path, goals=goals)
