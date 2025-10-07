import json
from typing import List, Literal, Optional, Union

from agents.schemas import AgentConfig, AgentDescription, AgentSpec, AgentType
from langchain.prompts import PromptTemplate
from langchain_core.tools import BaseTool, StructuredTool, tool

from convlab.e2e.multiwoz_dialogue_agent.state import FoodType, Price, Restaurant
from convlab.util.unified_datasets_util import load_database

from .system_prompt import RESTAURANT_AGENT_SYSTEM_INSTRUCTION


def end_conversation() -> str:
    """End the conversation with the user if the user explicitly wants to end the conversation. This does not end the conversation immediately, so you have a chance to say goodbye afterwards."""
    return "Conversation ended"


def query_restaurants(
    location: Optional[Literal["north", "south", "east", "west", "centre"]] = None,
    pricerange: Optional[Price] = None,
    food: Optional[FoodType] = None,
) -> str | None:
    """Query the restaurant database by various criteria."""
    database = load_database("multiwoz21")

    params = {
        "area": location,
        "pricerange": pricerange,
        "food": food,
    }

    query_params = {k: v for k, v in params.items() if v is not None}

    restaurants = database.query("restaurant", {"restaurant": query_params}, topk=1)
    if restaurants:
        print(restaurants)
        restaurant = Restaurant(**restaurants[0])
        print(restaurant)
        return json.dumps(restaurant.model_dump())
    
    return None



class RestaurantBookingAgent(AgentSpec):
    def __init__(self):
        pass

    @property
    def type(self) -> AgentType:
        return AgentType.RESTAURANT

    def get_config(self) -> Optional[AgentConfig]:
        return None

    def get_system_prompt(self) -> PromptTemplate:
        return RESTAURANT_AGENT_SYSTEM_INSTRUCTION

    def get_description(self) -> AgentDescription:
        return AgentDescription(
            description="Provides information about restaurants, can propose restaurants that satisfy the user's criteria and can book tables"
        )

    def get_tools(self) :
        return [query_restaurants, end_conversation]
