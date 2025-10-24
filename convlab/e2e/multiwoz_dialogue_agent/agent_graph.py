import json
import os
import sys
from typing import Any

from art.langgraph import init_chat_model as train_model
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver

from convlab.e2e.multiwoz_dialogue_agent.state import AgentState

load_dotenv()

from langchain.schema import AIMessage, HumanMessage
from langgraph.graph import MessagesState, StateGraph

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from agents.hotel.hotel_agent import HotelBookingAgent
from agents.restaurant.restaurant_agent import RestaurantBookingAgent
from agents.schemas import AgentType
from llm.utils.graph_util import populate_agent_graph
from llm.utils.langchain_client import create_openai_client

AGENT_GPT_MODEL_ID = "gpt-4o"
ROUTER_GPT_MODEL_ID = "gpt-4o-mini"
AGENT_MODEL_MAX_TOKENS = 512
ROUTER_MODEL_MAX_TOKENS = 16
AGENT_MODEL_TEMPERATURE = 0
ROUTER_MODEL_TEMPERATURE = 0
LOCALE = "en-US"

NUMBER_TO_WORDS = {
    "0": "null",
    "1": "eins",
    "2": "zwei",
    "3": "drei",
    "4": "vier",
    "5": "fünf",
    "6": "sechs",
    "7": "sieben",
    "8": "acht",
    "9": "neun",
}


def prepare_phone_number_for_synthesis(phone_number: str) -> str:
    """
    Prepare a phone number for speech synthesis by converting digits to German words.

    Args:
        phone_number: Phone number string

    Returns:
        str: Phone number with digits converted to German words, separated by commas
    """
    return ", ".join(
        NUMBER_TO_WORDS.get(char, char) for char in phone_number.replace(" ", "")
    )


def load_data_from_file(file_path: str) -> Any:
    """Load data from a JSON file."""
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return None


script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "scripts", "assets", "assistant.json")
assistant_config = load_data_from_file(file_path)


graph_state: AgentState = {
    "messages": [
        HumanMessage(
            content=f'[Nutzer mit der Rufnummer "{prepare_phone_number_for_synthesis("0123 1231234")}" ruft den Assistenten unter "{prepare_phone_number_for_synthesis("0321 3213210")}" telefonisch an]'
        ),
        AIMessage(content=assistant_config["assistant"]["welcomeMessage"]),
    ],
    "agent": None,
}

workflow = StateGraph(AgentState)


def get_workflow():
    """Create a new independent workflow instance each time."""
    new_workflow = StateGraph(AgentState)

    populate_agent_graph(
        workflow=new_workflow,
        config=assistant_config,
        locale=LOCALE,
        agents=[
            HotelBookingAgent(),
            RestaurantBookingAgent(),
        ],
        agent_client=train_model(name="dialogue_agent-agent-001", temperature=1.0),
        router_client=train_model(name="dialogue_agent-agent-001", temperature=1.0),
        # agent_client=create_openai_client(
        #     model=assistant_config.get("models", {})
        #     .get("chat", {})
        #     .get("model", AGENT_GPT_MODEL_ID),
        #     temperature=AGENT_MODEL_TEMPERATURE,
        #     max_tokens=AGENT_MODEL_MAX_TOKENS,
        # ),
        # router_client=create_openai_client(
        #     model=assistant_config.get("models", {})
        #     .get("router", {})
        #     .get("model", ROUTER_GPT_MODEL_ID),
        #     temperature=ROUTER_MODEL_TEMPERATURE,
        #     max_tokens=ROUTER_MODEL_MAX_TOKENS,
        # ),
    )

    return new_workflow


# populate_agent_graph(
#     workflow=workflow,
#     config=assistant_config,
#     locale=LOCALE,
#     agents=[
#         # IntentDiscoveryAgent(),
#         HotelBookingAgent(),
#         RestaurantBookingAgent(),
#     ],
#     agent_client=train_model(
#         # model=assistant_config.get("models", {})
#         # .get("chat", {})
#         # .get("model", AGENT_GPT_MODEL_ID),
#         # temperature=AGENT_MODEL_TEMPERATURE,
#         # max_tokens=AGENT_MODEL_MAX_TOKENS,
#     ),
#     router_client=train_model(
#         # model=assistant_config.get("models", {})
#         # .get("router", {})
#         # .get("model", ROUTER_GPT_MODEL_ID),
#         # temperature=ROUTER_MODEL_TEMPERATURE,
#         # max_tokens=ROUTER_MODEL_MAX_TOKENS,
#     ),
# )
#
# graph = workflow.compile()
