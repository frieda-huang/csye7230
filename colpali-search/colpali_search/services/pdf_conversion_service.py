from __future__ import annotations

import uuid
from itertools import islice
from typing import Dict, List, Optional

from colpali_search.schemas import ImageMetadata
from fastapi import UploadFile
from pdf2image import convert_from_bytes
from PIL import Image


class PDFConversionService:
    def __init__(
        self,
        pdf_files: List[bytes],
        images_list: Optional[List[List[Image.Image]]] = None,
        pdf_metadata: Optional[Dict[int, List[ImageMetadata]]] = None,
    ):
        self.pdf_files = pdf_files
        self.images_list = images_list
        self.pdf_metadata = pdf_metadata

    @staticmethod
    def generate_images_metadata(
        filename: str,
        total_pages: int,
    ) -> List[ImageMetadata]:

        return [
            ImageMetadata(
                page_number=page_number + 1,
                filename=filename,
                total_pages=total_pages,
            )
            for page_number in range(total_pages)
        ]

    @staticmethod
    def get_uuid() -> str:
        return str(uuid.uuid4())

    @classmethod
    def convert_pdfs2image(cls, pdf_files: List[UploadFile], batch_size=4):
        images_list = []
        pdf_metadata: Dict[str, List[ImageMetadata]] = {}

        def process_batch(batch):
            for pdf_file in batch:
                single_pdf_images = convert_from_bytes(pdf_file, thread_count=3)
                total_pages = len(single_pdf_images)
                pdf_id = cls.get_uuid()

                images_list.append(single_pdf_images)
                pdf_metadata[pdf_id] = cls.generate_images_metadata(
                    filename=pdf_file.filename,
                    total_pages=total_pages,
                )

        it = iter(pdf_files)
        while batch := list(islice(it, batch_size)):
            process_batch(batch)

        return cls(pdf_files, images_list, pdf_metadata)
