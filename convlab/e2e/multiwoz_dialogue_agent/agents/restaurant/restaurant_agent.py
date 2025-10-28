from typing import Literal, Optional

from agents.schemas import AgentConfig, AgentDescription, AgentSpec, AgentType
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool

from convlab.e2e.multiwoz_dialogue_agent.state import Booking, FoodType, Restaurant
from convlab.util.unified_datasets_util import load_database

from .system_prompt import RESTAURANT_AGENT_SYSTEM_INSTRUCTION


@tool
def end_conversation() -> str:
    """End the conversation with the user if the user explicitly wants to end the conversation. This does not end the conversation immediately, so you have a chance to say goodbye afterwards."""
    return "Conversation ended"


@tool
def book_table(
    restaurant_name: str,
    day: Literal[
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
    ],
    people: int,
    time: str,
) -> dict | str:
    """Tool to book a table at a restaurant.

    Args:
        restaurant_name: Name of the restaurant
        people: Number of people to book for
        time: At what time to book the table

    Returns:
        Booking record or failure message
    """
    database = load_database("multiwoz21")

    restaurants = database.query(
        "restaurant", {"restaurant": {"name": restaurant_name.lower()}}, topk=1
    )

    if restaurants:
        restaurant = Restaurant(**restaurants[0])
        booking = Booking(booking_number=restaurant.Ref)
        return booking.model_dump()

    return "Restaurant could not be found."


@tool(
    description="A database lookup for restaurants. Useful for when you need to check if there exist hotels that satisfy the user's criteria"
)
def search_restaurants(
    name: Optional[str] = None,
    location: Optional[Literal["north", "south", "east", "west", "centre"]] = None,
    pricerange: Optional[Literal["cheap", "moderate", "expensive"]] = None,
    food: Optional[FoodType] = None,
) -> dict | str:
    """Database lookup for restaurants. Useful for when you need to check if there exist hotels that satisfy the user's criteria

    Args:
        name: Name of the restaurant
        location: Location of the restaurant
        pricerange: Pricerange of the restaurant
        food: Type of food the restaurant serves

    Returns:
        Restaurant record or failure message
    """
    database = load_database("multiwoz21")

    params = {
        "name": name,
        "area": location,
        "pricerange": pricerange,
        "food": food,
    }

    query_params = {k: v for k, v in params.items() if v is not None}

    restaurants = database.query("restaurant", {"restaurant": query_params}, topk=1)
    if restaurants:
        restaurant = Restaurant(**restaurants[0])
        return restaurant.model_dump()

    return "There are no restaurants that satisfy the user's request parameters"


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

    def get_tools(self):
        return [search_restaurants, book_table, end_conversation]
