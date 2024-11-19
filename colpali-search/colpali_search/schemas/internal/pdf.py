from typing import Dict, List, TypeAlias

from PIL import Image
from pydantic import BaseModel, ConfigDict


class ImageMetadata(BaseModel):
    """Metadata on a single PDF Image"""

    filename: str
    page_number: int
    total_pages: int


ImageList: TypeAlias = List[List[Image.Image]]

PDFMetadata: TypeAlias = Dict[int, List[ImageMetadata]]


class SinglePDFConversion(BaseModel):
    pdf_id: int
    single_pdf_images: List[Image.Image]
    metadata: List[ImageMetadata]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PDFsConversion(BaseModel):
    pdf_files: List[bytes]
    images_list: ImageList
    pdf_metadata: PDFMetadata

    model_config = ConfigDict(arbitrary_types_allowed=True)
