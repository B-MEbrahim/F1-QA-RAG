from pydantic import BaseModel, Field
from typing import Any, Dict, Union, Optional

class Regulations(BaseModel):
    year: int = Field(2026, description="Year of FIA Regulations")

class Race(BaseModel):
    year: int = Field(2026, description="Race Year.")
    gp_name: str = Field(..., description="Name of Country where Race is held.")
