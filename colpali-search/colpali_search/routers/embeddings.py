from typing import List

from colpali_search.dependencies import (
    EmbeddingSerivceDep,
    ModelServiceDep,
    PDFConversionServiceDep,
)
from fastapi import APIRouter, Depends, HTTPException, UploadFile

embeddings_router = APIRouter(prefix="/embeddings", tags=["embeddings"])


async def validate_file_type(file: UploadFile) -> UploadFile:
    file_type = file.content_type
    if file_type != "application/pdf":
        raise HTTPException(
            status_code=404,
            detail=f"Invalid file type. Expected: 'application/pdf'. "
            f"Received: '{file_type}'. Please upload a valid PDF file.",
        )
    return file


@embeddings_router.post("/file")
async def generate_embeddings_for_file(
    model_service: ModelServiceDep,
    embedding_service: EmbeddingSerivceDep,
    pdf_conversion_service: PDFConversionServiceDep,
    pdf_file: UploadFile = Depends(validate_file_type),
):
    result = pdf_conversion_service.convert_pdfs2image([pdf_file])

    embeddings = model_service.embed_images(result.images_list, result.pdf_metadata)

    metadata = list(result.pdf_metadata.values())
    mdata = []
    mdata.extend(metadata)
    embedding_service.upsert_doc_embeddings(embeddings=embeddings, metadata=mdata)

    return {"filename": pdf_file.filename, "file_type": pdf_file.content_type}


@embeddings_router.post("/files")
async def generate_embeddings_for_files(
    files: List[UploadFile] = Depends(validate_file_type),
):
    pass


@embeddings_router.get("/status/{job_id}")
async def get_embedding_job_status(job_id: int):
    pass
