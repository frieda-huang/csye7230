from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import torch


@dataclass
class ImageMetadata:
    """Metadata on a single PDF Image"""

    pdf_id: str
    page_id: int
    filename: str
    total_pages: int
    filepath: str

    def to_json(self) -> Dict[str, Any]:
        """Convert relevant metadata to dict for storing embeddings."""
        metadata_dict = {
            "pdf_id": self.pdf_id,
            "page_id": self.page_id,
            "total_pages": self.total_pages,
            "filename": self.filename,
            "filepath": self.filepath,
        }
        return metadata_dict


@dataclass
class StoredImageData:
    embedding: torch.Tensor
    metadata: ImageMetadata
    created_at: str
    modified_at: str

    @classmethod
    def from_json(cls, json_data: Dict[str, Any]) -> StoredImageData:
        metadata = ImageMetadata(**json_data["metadata"])
        return cls(
            embedding=torch.tensor(json_data["embedding"]),
            metadata=metadata,
            created_at=json_data["created_at"],
            modified_at=json_data["modified_at"],
        )

    def to_json(self):
        return {
            "embedding": self.embedding,
            "metadata": self.metadata.to_json(),
            "created_at": self.created_at,
            "modified_at": self.modified_at,
        }
