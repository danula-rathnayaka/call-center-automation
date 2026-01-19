from typing import Literal

from pydantic import BaseModel, Field


class RouteResponse(BaseModel):
    intent: Literal["technical", "casual", "escalation"] = Field(
        ...,
        description="The classification of the user's query."
    )
