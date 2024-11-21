from colpali_search.database import async_session
from colpali_search.services.embedding_service import EmbeddingSerivce
from colpali_search.services.file_service import FileService
from colpali_search.services.model_service import ColPaliModelService
from colpali_search.services.pdf_conversion_service import PDFConversionService
from colpali_search.services.search_service import SearchService
from colpali_search.services.indexing_service import IndexingService


class AppContext:
    model_service: ColPaliModelService
    embedding_service: EmbeddingSerivce
    pdf_conversion_service: PDFConversionService
    search_service: SearchService
    file_service: FileService
    indexing_service: IndexingService


app_context = AppContext()


def initialize_context():
    app_context.model_service = ColPaliModelService()
    app_context.pdf_conversion_service = PDFConversionService()
    app_context.search_service = SearchService()
    app_context.indexing_service = IndexingService(async_session())
    app_context.embedding_service = EmbeddingSerivce(
        async_session(), indexing_service=app_context.indexing_service
    )
    app_context.file_service = FileService(async_session())
