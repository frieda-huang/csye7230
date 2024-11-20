import asyncio
from typing import List

from colpali_search.dependencies import (
    EmbeddingSerivceDep,
    ModelServiceDep,
    PDFConversionServiceDep,
)
from colpali_search.schemas.endpoints.embeddings import EmbeddingsFileResponse
from colpali_search.utils import generate_uuid
from fastapi import APIRouter, Depends, HTTPException, UploadFile

embeddings_router = APIRouter(prefix="/embeddings", tags=["embeddings"])


class FileValidator:
    allowed_types: List[str] = ["application/pdf"]

    @classmethod
    def _validate_file_type(cls, file: UploadFile) -> None:
        file_type = file.content_type
        if file_type not in cls.allowed_types:
            raise HTTPException(
                status_code=404,
                detail=f"Invalid file type. Expected: {cls.allowed_types}. "
                f"Received: '{file_type}'.",
            )

    @classmethod
    def validate_file(cls, file: UploadFile) -> UploadFile:
        cls._validate_file_type(file)
        return file

    @classmethod
    def validate_files(cls, files: List[UploadFile]) -> List[UploadFile]:
        for file in files:
            cls._validate_file_type(file)
        return files


@embeddings_router.post("/file")
async def generate_embeddings_for_file(
    model_service: ModelServiceDep,
    embedding_service: EmbeddingSerivceDep,
    pdf_conversion_service: PDFConversionServiceDep,
    file: UploadFile = Depends(FileValidator.validate_file),
) -> EmbeddingsFileResponse:
    try:
        loop = asyncio.get_running_loop()

        result = await loop.run_in_executor(
            None, pdf_conversion_service.convert_single_pdf2image, file
        )

        images, metadata = result.single_pdf_images, result.metadata
        embeddings = model_service.embed_images([images], [metadata])

        await embedding_service.upsert_doc_embeddings(
            embeddings=embeddings, metadata=metadata
        )

        return EmbeddingsFileResponse(
            id=generate_uuid(),
            embeddings=[tensor.view(-1).tolist() for tensor in embeddings],
            metadata=metadata,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during embedding: {str(e)}")


@embeddings_router.post("/files")
async def generate_embeddings_for_files(
    model_service: ModelServiceDep,
    embedding_service: EmbeddingSerivceDep,
    pdf_conversion_service: PDFConversionServiceDep,
    files: List[UploadFile] = Depends(FileValidator.validate_files),
) -> List[EmbeddingsFileResponse]:
    try:
        final_response = []
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, pdf_conversion_service.convert_pdfs2image, files
        )

        images_list, pdf_metadata = result.images_list, result.pdf_metadata

        for images, metadata in zip(images_list, pdf_metadata.values()):
            embeddings = model_service.embed_images([images], pdf_metadata)

            await embedding_service.upsert_doc_embeddings(
                embeddings=embeddings, metadata=metadata
            )
            final_response.append(
                EmbeddingsFileResponse(
                    id=generate_uuid(),
                    embeddings=[tensor.view(-1).tolist() for tensor in embeddings],
                    metadata=metadata,
                )
            )
        return final_response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during embedding: {str(e)}")


@embeddings_router.get("/status/{job_id}")
async def get_embedding_job_status(job_id: int):
    pass
