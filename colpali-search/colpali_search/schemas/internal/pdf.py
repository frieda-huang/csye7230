from typing import List, TypeAlias

from colpali_search.schemas.common import ImageMetadata
from PIL import Image
from pydantic import BaseModel, ConfigDict

ImageList: TypeAlias = List[List[Image.Image]]

MetadataList: TypeAlias = List[List[ImageMetadata]]


class SinglePDFConversion(BaseModel):
    pdf_id: str
    single_pdf_images: List[Image.Image]
    metadata: List[ImageMetadata]

    model_config = ConfigDict(arbitrary_types_allowed=True)


class PDFsConversion(BaseModel):
    images_list: ImageList
    metadata_list: MetadataList

    model_config = ConfigDict(arbitrary_types_allowed=True)
