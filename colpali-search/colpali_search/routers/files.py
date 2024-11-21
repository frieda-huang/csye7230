from typing import List

from colpali_search.dependencies import FileServiceDep
from colpali_search.schemas.endpoints.file import FileResultResponse
from fastapi import APIRouter, HTTPException

files_router = APIRouter(prefix="/files", tags=["files"])


@files_router.get("/")
async def get_all_files(
    file_service: FileServiceDep,
) -> List[FileResultResponse]:
    try:
        files = await file_service.get_all_files()
        return [FileResultResponse.model_validate(file) for file in files]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while retrieving files: {str(e)}",
        )


@files_router.get("/{id}")
async def get_file_by_id(id: int, file_service: FileServiceDep) -> FileResultResponse:
    try:
        file = await file_service.get_file_by_id(id)
        return FileResultResponse.model_validate(file)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while retrieving the file with ID {id}: {str(e)}",
        )


@files_router.delete("/{id}")
async def delete_file_by_id(id: int, file_service: FileServiceDep):
    try:
        await file_service.delete_file_by_id(id)
        return {"success": "Deleted"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred while deleting the file with ID {id}: {str(e)}",
        )
