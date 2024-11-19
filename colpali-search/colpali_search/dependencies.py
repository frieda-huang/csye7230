from colpali_search.database import async_session
from colpali_search.services.embedding_service import EmbeddingSerivce
from colpali_search.services.model_service import ColPaliModelService
from colpali_search.services.pdf_conversion_service import PDFConversionService
from colpali_search.services.search_service import SearchService


async def get_model_service() -> ColPaliModelService:
    return ColPaliModelService()


async def get_pdf_conversion_service() -> PDFConversionService:
    return PDFConversionService()


async def get_embedding_service() -> EmbeddingSerivce:
    return EmbeddingSerivce(session=async_session())


async def get_search_service() -> SearchService:
    return SearchService()
