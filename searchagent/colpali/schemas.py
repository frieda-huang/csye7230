from __future__ import annotations

import torch
from pydantic import BaseModel


class ImageMetadata(BaseModel):
    """Metadata on a single PDF Image"""

    pdf_id: str
    page_id: int
    filename: str
    total_pages: int
    filepath: str


class StoredImageData(BaseModel):
    embedding: torch.Tensor
    metadata: ImageMetadata
    created_at: str
    modified_at: str

    class Config:
        arbitrary_types_allowed = True
