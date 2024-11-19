from pydantic import BaseModel, EmailStr


class SearchRequest(BaseModel):
    query: str
    top_k: int
    email: EmailStr
