import json
import locale as locale_module
import traceback
from datetime import datetime
from enum import Enum
from typing import Callable, List, Literal, Optional

from art.langgraph.llm_wrapper import LoggingLLM
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    get_buffer_string,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START
from langgraph.graph.state import RunnableConfig
from langgraph.prebuilt import ToolNode
from langgraph.types import Command

from convlab.e2e.multiwoz_dialogue_agent.state import AgentState, RouteIntent

from ..prompts.formality_prompts import (
    ASSISTANT_FORMAL_PROMPT,
    ASSISTANT_INFORMAL_PROMPT,
)
from .router_system_prompt import ROUTER_SYSTEM_INSTRUCTION_TEMPLATE


class AssistantFormalityLevel(Enum):
    FORMAL = "FORMAL"
    INFORMAL = "INFORMAL"


DEFAULT_UNKNOWN_ASSISTANT_NAME = "unknown"
DEFAULT_BUSINESS_DESCRIPTION = "No further information."
SYSTEM_PROMPT_LOCALE = "en-US"

AGENT_TOOL_NODE_MAP = {
    "HOTEL": "HOTEL_TOOLS",
    "RESTAURANT": "RESTAURANT_TOOLS",
}

ROUTER_NODE = "AGENT_ROUTER"


def get_language_by_locale(locale: str) -> str:
    """Extract language from locale string"""
    return locale.split("-")[0] if "-" in locale else locale


def get_current_locale_date_time(locale_str: str) -> str:
    """
    Get current date and time formatted for the specified locale.

    Args:
        locale_str: Locale string (e.g., 'en-US', 'de-DE')

    Returns:
        str: Formatted date and time string
    """
    try:
        # Try to set the locale for formatting
        try:
            locale_module.setlocale(locale_module.LC_TIME, locale_str)
        except locale_module.Error:
            # Fallback to default locale if specified locale is not available
            pass

        now = datetime.now()
        # Format similar to JavaScript's toLocaleDateString with options
        formatted_date = now.strftime("%A, %B %d, %Y %H:%M")
        return formatted_date
    except Exception:
        # Fallback to ISO format if locale formatting fails
        return datetime.now().strftime("%Y-%m-%d %H:%M")


def create_router_output_schema(agent_values: List[str]):
    """
    Create a simple schema for router output validation.

    Args:
        agent_values: List of valid agent type strings

    Returns:
        dict: Schema definition for router output
    """
    return {
        "title": "RouterOutput",
        "description": "Output schema for the router node",
        "type": "object",
        "properties": {
            "agent": {
                "type": "string",
                "enum": agent_values,
                "description": "Specifies the agent that should handle the user's request.",
            }
        },
        "required": ["agent"],
    }


def populate_agent_graph(
    workflow,
    config,
    agents: List,
    agent_client,
    router_client,
    locale: str,
    on_error: Optional[Callable[[Exception], None]] = None,
):
    """
    Populate an agent graph with nodes and edges for workflow processing.

    Args:
        workflow: AgentStateGraph instance
        config: AssistantConfig instance
        agents: List of Agent instances
        agent_client: ChatClient instance for agents
        router_client: ChatClient instance for router
        locale: Language/locale string
        on_error: Optional error handler function

    Returns:
        AgentStateGraph: The populated workflow graph
    """
    for agent in agents:
        workflow.add_node(
            agent.type.value,
            create_agent_node(
                config=config,
                agent=agent,
                model=agent_client,
                locale=locale,
                on_error=on_error,
            ),
        ).add_node(
            AGENT_TOOL_NODE_MAP[agent.type.value], ToolNode(agent.get_tools())
        ).add_edge(
            AGENT_TOOL_NODE_MAP[agent.type.value], agent.type.value
        ).add_conditional_edges(
            agent.type.value,
            create_agent_output_edge(),
            [AGENT_TOOL_NODE_MAP[agent.type.value], END],
        )

    workflow.add_node(
        ROUTER_NODE,
        create_agent_router_node(model=router_client, agents=agents, on_error=on_error),
    ).add_edge(START, ROUTER_NODE).add_conditional_edges(
        ROUTER_NODE, create_router_edge(), [agent.type.value for agent in agents]
    )

    return workflow


def create_agent_output_edge():

    def converse_with_user(state: AgentState, config: RunnableConfig) -> str:
        messages = state["messages"]
        last_message = messages[-1] if messages else None

        if (
            last_message
            and isinstance(last_message, AIMessage)
            and hasattr(last_message, "tool_calls")
            and last_message.tool_calls
            and len(last_message.tool_calls) > 0
        ):
            has_end_conversation = any(
                tool_call.get("name") == "end_conversation"
                for tool_call in last_message.tool_calls
            )

            if has_end_conversation:
                for message in reversed(messages):
                    if (
                        isinstance(message, AIMessage)
                        and hasattr(message, "tool_calls")
                        and message.tool_calls
                        and len(message.tool_calls) > 0
                    ):
                        for tool_call in message.tool_calls:
                            if tool_call.get("name") != "end_conversation":
                                break
                        else:
                            continue
                        break

            return AGENT_TOOL_NODE_MAP[state["agent"].value]

        return END

    return converse_with_user


def create_agent_router_node(
    model: LoggingLLM,
    agents: List,
    on_error: Optional[Callable[[Exception], None]] = None,
) -> Callable:

    def route_intent(state: AgentState, config: RunnableConfig) -> Command:
        try:
            # agent_types = [agent.type.value for agent in agents]

            agent_descriptions = [
                {"agent": agent.type.value, "description": agent.get_description()}
                for agent in agents
            ]

            # output_schema = create_router_output_schema(agent_types)

            model_with_output = model.with_structured_output(RouteIntent)

            agents_json = json.dumps(agent_descriptions, indent=2)

            prompt_content = ROUTER_SYSTEM_INSTRUCTION_TEMPLATE.format(
                agents=agents_json, messages=get_buffer_string(state["messages"])
            )

            result = model_with_output.with_retry(stop_after_attempt=3).invoke(
                [SystemMessage(content=prompt_content)]
            )

            if result.intent == "HOTEL":
                return Command(
                    goto=result.intent,
                    update={"agent": result.intent},
                )

            if result.intent == "RESTAURANT":
                return Command(
                    goto=result.intent,
                    update={"agent": result.intent},
                )

            return {"agent": result.intent}

        except Exception as error:
            if on_error:
                print(f"Router failed")
            return Command(goto="END")

    return route_intent


def create_router_edge():

    def router_edge_handler(state):
        return state["agent"]

    return router_edge_handler


def create_agent_node(
    config,
    agent,
    model: LoggingLLM,
    locale: str,
    on_error: Optional[Callable[[Exception], None]] = None,
):
    """
    Create an agent node function for workflow processing.

    Args:
        config: AssistantConfig instance
        agent: Agent instance
        model: ChatClient instance
        locale: Language/locale string
        on_error: Optional error handler function

    Returns:
        Callable: Function that processes agent node state
    """

    def agent_node_handler(state: AgentState) -> dict:
        try:
            format_kwargs = {
                "assistantName": config["assistant"]["name"]
                or DEFAULT_UNKNOWN_ASSISTANT_NAME,
                "assistantFormalityPrompt": (
                    ASSISTANT_INFORMAL_PROMPT
                    if config["assistant"]["formalityLevel"]
                    == AssistantFormalityLevel.INFORMAL
                    else ASSISTANT_FORMAL_PROMPT
                ),
                # "primaryDomain": config["assistant"]["primaryDomain"]
                "businessName": config["business"]["name"],
                "businessType": config["business"]["type"],
                "businessSector": config["business"]["sector"],
                "businessDescription": DEFAULT_BUSINESS_DESCRIPTION,
                "dateTime": get_current_locale_date_time(SYSTEM_PROMPT_LOCALE),
                "language": "English",
                "messages": get_buffer_string(state["messages"]),
            }

            system_prompt = agent.get_system_prompt().format(**format_kwargs)

            response = model.bind_tools(agent.get_tools()).invoke(
                [SystemMessage(system_prompt)] + state.get("messages", [])
            )

            # print(f"{state=}")

            belief_state = state["belief_state"]

            last_message = state["messages"][-1]

            if last_message and isinstance(last_message, ToolMessage):
                if isinstance(last_message.content, str):
                    try:
                        tool_content = json.loads(last_message.content)

                        if last_message.name in ["search_hotels", "search_restaurants"]:
                            db_entity = tool_content
                            active_domain = (
                                state["agent"].value.lower()
                                if state["agent"]
                                else "unknown"
                            )

                            for slot, db_value in db_entity.items():
                                if slot in belief_state[active_domain]:
                                    if belief_state[active_domain][slot] != db_value:
                                        belief_state[active_domain][slot] = db_value
                                else:
                                    belief_state[active_domain][slot] = db_value

                    except (json.JSONDecodeError, KeyError):
                        pass

            if (
                isinstance(response, AIMessage)
                and hasattr(response, "tool_calls")
                and response.tool_calls
            ):
                if response.tool_calls[0].get("name") == "book_hotel":
                    tool_call = response.tool_calls[0]
                    arguments = tool_call.get("args", {})

                    day = arguments.get("day")
                    people = arguments.get("people")
                    n_nights = arguments.get("n_nights")

                    belief_state["hotel"]["book day"] = str(day)
                    belief_state["hotel"]["book people"] = str(people)
                    belief_state["hotel"]["book stay"] = str(n_nights)

                if response.tool_calls[0].get("name") == "book_table":
                    tool_call = response.tool_calls[0]
                    arguments = tool_call.get("args", {})

                    day = arguments.get("day")
                    people = arguments.get("people")
                    time = arguments.get("time")

                    belief_state["restaurant"]["book day"] = str(day)
                    belief_state["restaurant"]["book people"] = str(people)
                    belief_state["restaurant"]["book time"] = str(time)

            return Command(
                goto=END, update={"messages": [response], "agent": agent.type}
            )

        except Exception as error:
            print(f"Error in agent node {agent.type}: {error}")
            print(f"Error type: {type(error).__name__}")
            print(f"Error details: {str(error)}")

            if on_error:
                on_error(error)
            return {"messages": [], "agent": agent.type}

    return agent_node_handler
