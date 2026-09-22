from pydantic import BaseModel, Field


class RAGResponse(BaseModel):
    respuesta: str = Field(min_length=1)
    fuentes: list[str] = Field(default_factory=list)