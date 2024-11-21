import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict


class FileResultResponse(BaseModel):
    id: int
    filename: str
    filetype: str
    total_pages: int
    summary: Optional[str]
    last_modified: datetime.datetime
    created_at: datetime.datetime

    model_config = ConfigDict(from_attributes=True)
