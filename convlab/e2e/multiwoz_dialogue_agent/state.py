from typing import Dict, Literal, Tuple

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

from convlab.e2e.multiwoz_dialogue_agent.agents.schemas import AgentType


class AgentState(MessagesState):
    agent: AgentType | None
    belief_state: Dict[str, Dict[str, str]]


class RouteIntent(BaseModel):
    """Routing decision for user requests"""

    intent: Literal["HOTEL", "RESTAURANT"] = Field(
        description="Target agent to handle the request"
    )


class Price(BaseModel):
    single: str | None = None
    double: str | None = None
    family: str | None = None


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
    Ref: str


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
    "spanish",
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
    introduction: str | None = None
    location: tuple[float, float]
    phone: str | None = None
    postcode: str
    pricerange: Literal["cheap", "moderate", "expensive"]
    type: Literal["restaurant"]
    signature: str | None = None
    Ref: str


class Booking(BaseModel):
    booking_number: str
