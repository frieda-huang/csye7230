from fastapi import APIRouter

files_router = APIRouter(prefix="/files", tags=["files"])


@files_router.get("/")
async def get_all_files():
    pass


@files_router.get("/{filepath}")
async def get_file_by_path():
    pass


@files_router.delete("/{filepath}")
async def delete_file_by_path():
    pass
