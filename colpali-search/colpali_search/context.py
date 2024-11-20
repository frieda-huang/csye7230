from colpali_search.database import async_session
from colpali_search.services.embedding_service import EmbeddingSerivce
from colpali_search.services.model_service import ColPaliModelService
from colpali_search.services.pdf_conversion_service import PDFConversionService
from colpali_search.services.search_service import SearchService


class AppContext:
    model_service: ColPaliModelService
    embedding_service: EmbeddingSerivce
    pdf_conversion_service: PDFConversionService
    search_service: SearchService


app_context = AppContext()


def initialize_context():
    app_context.model_service = ColPaliModelService()
    app_context.embedding_service = EmbeddingSerivce(async_session())
    app_context.pdf_conversion_service = PDFConversionService()
    app_context.search_service = SearchService()
