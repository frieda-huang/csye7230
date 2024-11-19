from __future__ import annotations

import uuid
from itertools import islice
from typing import Dict, List

from colpali_search.schemas.internal.pdf import (
    ImageMetadata,
    PDFsConversion,
    SinglePDFConversion,
)
from fastapi import UploadFile
from pdf2image import convert_from_bytes


class PDFConversionService:
    def _generate_images_metadata(
        self,
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

    def convert_single_pdf2image(self, pdf_file: UploadFile) -> SinglePDFConversion:
        single_pdf_images = convert_from_bytes(pdf_file, thread_count=3)
        total_pages = len(single_pdf_images)
        pdf_id = self.get_uuid()
        metadata = self._generate_images_metadata(
            filename=pdf_file.filename,
            total_pages=total_pages,
        )
        return single_pdf_images, pdf_id, metadata

    def convert_pdfs2image(
        self, pdf_files: List[UploadFile], batch_size=4
    ) -> PDFsConversion:
        images_list = []
        pdf_metadata: Dict[str, List[ImageMetadata]] = {}

        def process_batch(batch):
            for pdf_file in batch:
                result = self.convert_single_pdf2image(pdf_file)
                images_list.append(result.single_pdf_images)
                pdf_metadata[result.pdf_id] = result.metadata

        it = iter(pdf_files)
        while batch := list(islice(it, batch_size)):
            process_batch(batch)

        return PDFsConversion(pdf_files, images_list, pdf_metadata)
