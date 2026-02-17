from pydantic import BaseModel, Field
from typing import Literal, Union

class Regulations(BaseModel):
    year: int = Field(2026, description="Year of FIA Regulations")


class Race(BaseModel):
    year: int = Field(2026, description="Race Year.")
    gp_name: str = Field(..., description="Name of Country where Race is held.")


class RouteQuery(BaseModel):
    """
    Route the user query to the most appropriate tool.
    """
    intent: Literal["REGULATIONS", "RACE_RESULTS", "GENERAL_CHAT"] = Field(
        ..., 
        description="The intent of the user. regulations=rules/specs, race_results=winners/podiums/standings, general_chat=other"
    )
    regulations_query: Union[Regulations, None] = Field(
        None,
        description="If intent is REGULATIONS, extract the year here."
    )
    race_query: Union[Race, None] = Field(
        None,
        description="If intent is RACE_RESULTS, extract year and gp_name here."
    )


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    question: str = Field(..., description="The user's question")
    session_id: str = Field(default="default", description="Session ID for chat history")


class ChatResponse(BaseModel):
    """Response model for simple chat endpoint."""
    answer: str


class FullResponse(BaseModel):
    """Response model for full ask endpoint."""
    answer: str
    intent: str
    sources: list
    validation_info: dict


class ClearRequest(BaseModel):
    """Request model for clear history endpoint."""
    session_id: str = Field(default="default")