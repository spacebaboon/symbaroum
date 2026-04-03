from typing import Literal
from pydantic import BaseModel


class QueryRequest(BaseModel):
    query: str


class QueryMetadata(BaseModel):
    pipeline: Literal["hybrid", "lightrag"]
    timings: dict[str, float]
    total: float