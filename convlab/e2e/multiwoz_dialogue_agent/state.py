from typing import Literal, Optional

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from convlab.e2e.multiwoz_dialogue_agent.agents.schemas import AgentType


class AgentState(MessagesState):
    agent: Optional[AgentType]


class RouteIntent(BaseModel):
    """Model for routing user requests"""

    intent: Literal["HOTEL", "RESTAURANT"] = Field(
        description="The name of the expert agent to route the request to"
    )
