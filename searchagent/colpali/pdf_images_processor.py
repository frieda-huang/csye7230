import os
import tempfile
from itertools import islice
from typing import Dict, List, Optional

from pdf2image import convert_from_path
from PIL import Image
from searchagent.colpali.models import ImageMetadata
from searchagent.file_system.base import FileSystemManager


class PDFImagesProcessor:
    def __init__(
        self,
        input_dir: Optional[str] = None,
        images_list: Optional[List[List[Image.Image]]] = None,
        pdf_metadata: Optional[Dict[int, List[ImageMetadata]]] = None,
    ):
        self.input_dir = input_dir
        self.images_list = images_list
        self.pdf_metadata = pdf_metadata

    @staticmethod
    def convert_single_pdf(filepath: str) -> List[Image.Image]:
        with tempfile.TemporaryDirectory() as path:
            return convert_from_path(filepath, output_folder=path, thread_count=3)

    @staticmethod
    def generate_images_metadata(
        pdf_id: str,
        filepath: str,
        total_pages: int,
    ) -> List[ImageMetadata]:
        filename = os.path.basename(filepath)

        return [
            ImageMetadata(
                pdf_id=pdf_id,
                page_id=page_id + 1,  # Start at indx 1
                filename=filename,
                total_pages=total_pages,
                filepath=filepath,
            )
            for page_id in range(total_pages)
        ]

    @classmethod
    def convert_pdf2image_from_dir(cls, input_dir: str, batch_size=4):
        import uuid

        filepaths = FileSystemManager(input_dir).list_files(["application/pdf"])

        images_list = []
        pdf_metadata: Dict[str, List[ImageMetadata]] = {}

        def process_batch(batch):
            for fp in batch:
                single_pdf_images = cls.convert_single_pdf(fp)
                total_pages = len(single_pdf_images)
                pdf_id = str(uuid.uuid4())

                images_list.append(single_pdf_images)
                pdf_metadata[pdf_id] = cls.generate_images_metadata(
                    pdf_id=pdf_id,
                    filepath=fp,
                    total_pages=total_pages,
                )

        it = iter(filepaths)
        while batch := list(islice(it, batch_size)):
            process_batch(batch)

        return cls(input_dir, images_list, pdf_metadata)
