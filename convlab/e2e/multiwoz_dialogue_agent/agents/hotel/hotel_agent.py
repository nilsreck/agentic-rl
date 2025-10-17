from typing import List, Literal, Optional, Union

from agents.schemas import AgentConfig, AgentDescription, AgentSpec, AgentType
from langchain.prompts import PromptTemplate
from langchain_core.tools import BaseTool, StructuredTool, tool

from convlab.e2e.multiwoz_dialogue_agent.state import Area, Booking, Hotel
from convlab.util.unified_datasets_util import load_database

from .system_prompt import HOTEL_AGENT_SYSTEM_INSTRUCTION


@tool
def end_conversation() -> str:
    """End the conversation with the user if the user explicitly wants to end the conversation. This does not end the conversation immediately, so you have a chance to say goodbye afterwards."""
    return "Conversation ended"


@tool
def book_hotel(
    day: Literal[
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"
    ],
    people: int,
    n_nights: int,
) -> dict:
    "Make a reservation that returns a made-up booking number"
    booking = Booking(booking_number="00000011")
    return booking.model_dump()


@tool
def search_hotels(
    name: Optional[str] = None,
    location: Optional[Area] = None,
    pricerange: Optional[Literal["cheap", "moderate", "expensive"]] = None,
    stars: Optional[Literal["0", "2", "3", "4"]] = None,
    hotel_type: Optional[Literal["hotel", "guesthouse"]] = None,
    internet: Optional[Literal["yes", "no"]] = None,
    parking: Optional[Literal["yes", "no"]] = None,
    takesbookings: Optional[Literal["yes", "no"]] = None,
) -> dict | str:
    """Query the hotel database by various criteria."""
    database = load_database("multiwoz21")

    params = {
        "name": name,
        "area": location,
        "pricerange": pricerange,
        "stars": stars,
        "type": hotel_type,
        "internet": internet,
        "parking": parking,
        "takesbookings": takesbookings,
    }

    query_params = {k: v for k, v in params.items() if v is not None}

    hotels = database.query("hotel", {"hotel": query_params}, topk=1)

    if hotels:
        hotel = Hotel(**hotels[0])
        return hotel.model_dump()

    return "There are no hotels that satisfy the user's request parameters"


QUERY_HOTELS: BaseTool = search_hotels
BOOKING_TOOL: BaseTool = book_hotel
END_CONVERSATION_TOOL: BaseTool = end_conversation


class HotelBookingAgent(AgentSpec):
    def __init__(self):
        pass

    @property
    def type(self) -> AgentType:
        return AgentType.HOTEL

    def get_config(self) -> Optional[AgentConfig]:
        return None

    def get_system_prompt(self) -> PromptTemplate:
        return HOTEL_AGENT_SYSTEM_INSTRUCTION

    def get_description(self) -> AgentDescription:
        return AgentDescription(
            description="Provides information about hotels, is able to search for hotels that satisfy the user's criteria, and can make hotel reservations"
        )

    def get_tools(self) -> List[Union[StructuredTool, BaseTool]]:
        return [END_CONVERSATION_TOOL, QUERY_HOTELS, BOOKING_TOOL]
