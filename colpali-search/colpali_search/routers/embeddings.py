from typing import List

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
async def generate_embeddings_for_file(file: UploadFile = Depends(validate_file_type)):
    return {"filename": file.filename, "file_type": file.content_type}


@embeddings_router.post("/files")
async def generate_embeddings_for_files(
    files: List[UploadFile] = Depends(validate_file_type),
):
    pass


@embeddings_router.get("/status/{job_id}")
async def get_embedding_job_status(job_id: int):
    pass
