import datetime
from typing import List, Optional

from pydantic import BaseModel, EmailStr


class SearchResult(BaseModel):
    file_id: int
    page_number: int
    filename: str
    filetype: str
    total_pages: int
    summary: Optional[str]
    last_modified: datetime.datetime
    created_at: datetime.datetime


class SearchRequest(BaseModel):
    query: str
    top_k: int
    email: EmailStr


class SearchResponse(BaseModel):
    result: List[SearchResult]
