from typing import Dict, List, TypeAlias

from colpali_search.schemas.common import ImageMetadata
from PIL import Image
from pydantic import BaseModel, ConfigDict

ImageList: TypeAlias = List[List[Image.Image]]

PDFMetadata: TypeAlias = Dict[int, List[ImageMetadata]]


class SinglePDFConversion(BaseModel):
    pdf_id: str
    single_pdf_images: List[Image.Image]
    metadata: List[ImageMetadata]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PDFsConversion(BaseModel):
    pdf_files: List[bytes]
    images_list: ImageList
    pdf_metadata: PDFMetadata

    model_config = ConfigDict(arbitrary_types_allowed=True)
