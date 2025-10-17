from typing import Dict, Literal, Optional, Tuple

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from convlab.e2e.multiwoz_dialogue_agent.agents.schemas import AgentType


class AgentState(MessagesState):
    agent: Optional[AgentType]
    belief_state: Dict[str, Dict[str, str]]


class RouteIntent(BaseModel):
    """Model for routing user requests"""

    intent: Literal["HOTEL", "RESTAURANT"] = Field(
        description="The name of the expert agent to route the request to"
    )


class Price(BaseModel):
    single: str
    double: str
    family: str


Area = Literal["north", "east", "west", "south", "centre"]


class Hotel(BaseModel):
    id: str
    name: str
    address: str
    area: Area
    internet: Literal["yes", "no"]
    parking: Literal["yes", "no"]
    location: Tuple[float, float]
    phone: str
    postcode: str
    price: Price
    pricerange: Literal["cheap", "moderate", "expensive"]
    stars: Literal["0", "2", "3", "4"]
    takesbookings: Literal["yes", "no"]
    type: Literal["hotel", "guesthouse"]


FoodType = Literal[
    "italian",
    "international",
    "indian",
    "chinese",
    "modern european",
    "european",
    "british",
    "mexican",
    "gastropub",
    "vietnamese",
    "french",
    "lebanese",
    "japanese",
    "korean",
    "turkish",
    "asian oriental",
    "african",
    "mediterranean",
    "north american",
    "seafood",
    "portuguese",
    "thai",
]


class Restaurant(BaseModel):
    id: str
    name: str
    address: str
    area: Area
    food: FoodType
    introduction: Optional[str] = None
    location: tuple[float, float]
    phone: Optional[str] = None
    postcode: str
    pricerange: Literal["cheap", "moderate", "expensive"]
    type: Literal["restaurant"]
    signature: Optional[str] = None


class Booking(BaseModel):
    booking_number: Literal["00000000", "00000011"]
