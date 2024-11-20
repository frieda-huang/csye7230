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


async def validate_file_type(pdf_file: UploadFile) -> UploadFile:
    file_type = pdf_file.content_type
    if file_type != "application/pdf":
        raise HTTPException(
            status_code=404,
            detail=f"Invalid file type. Expected: 'application/pdf'. "
            f"Received: '{file_type}'. Please upload a valid PDF file.",
        )
    return pdf_file


@embeddings_router.post("/file")
async def generate_embeddings_for_file(
    model_service: ModelServiceDep,
    embedding_service: EmbeddingSerivceDep,
    pdf_conversion_service: PDFConversionServiceDep,
    pdf_file: UploadFile = Depends(validate_file_type),
) -> EmbeddingsFileResponse:
    try:
        loop = asyncio.get_running_loop()

        result = await loop.run_in_executor(
            None, pdf_conversion_service.convert_single_pdf2image, pdf_file
        )

        images, metadata = result.single_pdf_images, result.metadata
        embeddings = model_service.embed_images([images], {0: images})

        await embedding_service.upsert_doc_embeddings(
            embeddings=embeddings, metadata=metadata
        )

        return EmbeddingsFileResponse(
            id=generate_uuid(),
            embeddings=[embed.flatten().tolist() for embed in embeddings],
            metadata=metadata,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during embedding: {str(e)}")


@embeddings_router.post("/files")
async def generate_embeddings_for_files(
    files: List[UploadFile] = Depends(validate_file_type),
):
    pass


@embeddings_router.get("/status/{job_id}")
async def get_embedding_job_status(job_id: int):
    pass
