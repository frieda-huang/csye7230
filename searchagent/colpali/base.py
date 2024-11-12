import asyncio
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import ml_dtypes
import numpy as np
import torch
from colpali_engine.models import ColPali
from colpali_engine.models.paligemma.colpali.processing_colpali import ColPaliProcessor
from colpali_engine.utils.torch_utils import ListDataset, get_torch_device
from PIL import Image
from searchagent.colpali.pdf_images_dataset import PDFImagesDataset
from searchagent.colpali.pdf_images_processor import PDFImagesProcessor
from searchagent.colpali.profiler import profile_colpali
from searchagent.colpali.repositories.all import (
    EmbeddingRepository,
    FileRepository,
    FlattenedEmbeddingRepository,
    FolderRepository,
    PageRepository,
    QueryRepository,
)
from searchagent.colpali.schemas import ImageMetadata
from searchagent.colpali.search_engine.context import IndexingContext, SearchContext
from searchagent.colpali.search_engine.strategy_factory import (
    IndexingStrategyFactory,
    SearchStrategyFactory,
)
from searchagent.db_connection import async_session, execute_analyze
from searchagent.models import Embedding, File, FlattenedEmbedding, Folder, Page, Query
from sqlalchemy.ext.asyncio import AsyncSession
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BatchFeature, PreTrainedModel


class ColPaliRag:
    def __init__(
        self,
        input_dir: Optional[Union[Path, str]] = None,
        model_name: Optional[str] = None,
        hf_api_key: Optional[str] = None,
        benchmark: bool = False,
    ):
        """
        A RAG system using a visual language model called Colpali to process PDF files

        Args:
            query (str): search query
            input_dir (Optional[Union[Path, str]]): Path to the PDF directory
            model_name (Optional[str]): Name of a Colpali pretrained model
                Default is "vidore/colpali-v1.2"
            benchmark (bool): Allow us to benchmark the performance of ColPali RAG system
        """

        # Lazy loadings
        self._model = None
        self._processor: Optional[ColPaliProcessor] = None
        self.user_id = 1  # Replace with actual user ID
        self.metadata = None

        self.device = get_torch_device()
        self.model_name = model_name or "vidore/colpali-v1.2"
        self.hf_api_key = hf_api_key or os.getenv("HF_API_KEY")
        self.benchmark = benchmark

        if input_dir:
            self.input_dir = Path(input_dir)

        if not self.hf_api_key:
            raise ValueError("HuggingFace API key is required")

        if input_dir and not self.input_dir.is_dir():
            raise ValueError(f"Input directory does not exist: {input_dir}")

        if not input_dir and not benchmark:
            raise ValueError("input_dir is required if benchmark mode is not on")

        if input_dir and benchmark:
            raise ValueError(
                "Benchmark mode is currently on. We will use ViDoRe dataset instead"
            )

        if input_dir:
            self.pdf_processor = PDFImagesProcessor.convert_pdf2image_from_dir(
                str(self.input_dir)
            )

        if not input_dir and benchmark:
            self.pdf_processor = PDFImagesProcessor.retrieve_pdfImage_from_vidore()

        self.images = self.pdf_processor.images_list
        self.pdf_metadata = self.pdf_processor.pdf_metadata

    @property
    def model(self) -> PreTrainedModel:
        if self._model is None:
            self._model = ColPali.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                token=self.hf_api_key,
            ).eval()
        return self._model

    @property
    def processor(self) -> ColPaliProcessor:
        """Get the ColPaliProcessor instance"""
        if self._processor is None:
            self._processor = ColPaliProcessor.from_pretrained(
                self.model_name, token=self.hf_api_key
            )
        return self._processor

    @torch.no_grad()
    def image_collate_fn(
        self, batch: List[Tuple[Image.Image, ImageMetadata]]
    ) -> Tuple[BatchFeature, List[ImageMetadata]]:
        """Given a batch size of 4, it would return a list of 4 tuples and
        unpack the batch into a list of images

        `batch` before unpacking:
            [(Image1, ImageMetadata1), (Image2, ImageMetadata2), ...]

        `batch` after unpacking:
            [Image1, Image2, Image3, ...]
        """

        images, metadata = zip(*batch)

        images: List[Image.Image] = list(images)
        metadata: List[ImageMetadata] = list(metadata)

        batch_images = self.processor.process_images(images).to(self.device)
        return batch_images, metadata

    def _create_dataloader(
        self,
        dataset: Union[PDFImagesDataset, List[str]],
        process_fn: Union[
            Callable[[List[str]], torch.Tensor],
            Callable[
                [Tuple[Image.Image, ImageMetadata]],
                Tuple[BatchFeature, List[ImageMetadata]],
            ],
        ],
        batch_size: int = 4,  # 4 PDF image pages
    ) -> DataLoader[str]:
        if isinstance(dataset, List) and dataset and isinstance(dataset[0], str):
            dataset = ListDataset[str](dataset)

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=process_fn,
        )

    @profile_colpali(enable_profiling=False)
    def run_inference(self, batches: dict):
        return self.model(**batches)

    # @measure_latency_for_gpu(dummy_input_fn=lambda: create_dummy_input())
    @torch.no_grad()
    def _embed(
        self,
        dataset: Union[PDFImagesDataset, List[str]],
        process_fn: Union[
            Callable[[List[str]], torch.Tensor],
            Callable[
                [List[Tuple[Image.Image, ImageMetadata]]],
                Tuple[BatchFeature, List[ImageMetadata]],
            ],
        ],
    ) -> List[torch.Tensor]:
        """Compute embeddings for either queries or images

        Args:
            dataset (Union[PDFImagesDataset, List[str]]): Images or queries to be processed
            process_fn (Union[
                Callable[[List[str]], torch.Tensor],
                Callable[
                    [List[Tuple[Image.Image, ImageMetadata]]],
                    Tuple[BatchFeature, List[ImageMetadata]],
                ]
            ):
                Either self.image_collate_fn or self.processor.process_queries

        Returns:
            List[torch.Tensor]: A list of computed embeddings
        """
        if process_fn not in [
            self.image_collate_fn,
            self.processor.process_queries,
        ]:
            raise ValueError(
                "process_fn must be either processor.process_images or processor.process_queries"
            )

        empty_lists = [[] for _ in range(2)]
        embeddings, mdata = empty_lists

        dataloader = self._create_dataloader(dataset, process_fn)

        for batch in tqdm(dataloader):
            if isinstance(dataset, PDFImagesDataset):
                batch_images, metadata = batch
                batches = {k: v.to(self.device) for k, v in batch_images.items()}
                mdata.extend(meta.model_dump() for meta in metadata)
                self.metadata = mdata
            else:
                batches = {k: v.to(self.device) for k, v in batch.items()}

            # Move images to GPU and get embeddings
            batch_embeddings = self.run_inference(batches)

            # Embeddings for a single PDF page; it is of the shape (1030, 128)
            # 32x32=1024 image patches and 6 instruction text tokens
            individual_embeddings = list(torch.unbind(batch_embeddings.to("cpu")))

            # Collect all embeddings
            embeddings.extend(individual_embeddings)

        return embeddings

    # @measure_vram()
    def embed_images(self) -> List[torch.Tensor]:
        # TODO: Set up a Cron job to sync files and their embeddings each night
        # TODO: Dynamically identify which PDF images need to be indexed
        """Embed images using custom Dataset
        Check this link on PyTorch custom Dataset:
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        """
        processed_images = PDFImagesDataset(self.images, self.pdf_metadata)
        return self._embed(processed_images, self.image_collate_fn)

    # @measure_vram()
    def embed_query(self, query: str) -> List[torch.Tensor]:
        return self._embed([query], self.processor.process_queries)

    def get_scores(self, qs: List[torch.Tensor], embeddings: List[torch.Tensor]):
        return self.processor.score(qs, embeddings)

    async def upsert_doc_embeddings(
        self, session: AsyncSession, embeddings: List[torch.Tensor]
    ):
        """Upsert embeddings to PostgreSQL"""

        folder_repo = FolderRepository(Folder, session=session)
        file_repo = FileRepository(File, session=session)
        page_repo = PageRepository(Page, session=session)
        embedding_repo = EmbeddingRepository(Embedding, session=session)
        flattened_embedding_repo = FlattenedEmbeddingRepository(
            FlattenedEmbedding, session=session
        )

        async with session.begin():

            chunk_size = 500
            embed_len = len(embeddings)

            for i in range(0, embed_len, chunk_size):

                chunk_embeddings = embeddings[i : i + chunk_size]
                chunk_metadata = self.metadata[i : i + chunk_size]

                for embedding, metadata in zip(chunk_embeddings, chunk_metadata):
                    page_id = metadata["page_id"]
                    embeddings = embedding.tolist()
                    vector_embedding = [np.array(e) for e in embeddings]
                    filename = metadata["filename"]
                    filepath = metadata["filepath"]
                    total_pages = metadata["total_pages"]
                    folder_name, folder_path = "", ""
                    folder, file = None, None

                    if not self.benchmark:
                        folder_name = str(self.input_dir.name)
                        folder_path = str(self.input_dir)

                    # Check if folder already exists
                    folder_entry_exists = (
                        await folder_repo.get_by_folder_path(str(self.input_dir))
                        if not self.benchmark
                        else False
                    )
                    if not folder_entry_exists:
                        folder = await folder_repo.add(
                            folder_name, folder_path, self.user_id
                        )
                    else:
                        folder = folder_entry_exists

                    # Check if file already exists
                    file_entry_exists = await file_repo.get_by_filepath_filename(
                        filepath, filename
                    )

                    if not file_entry_exists:
                        file = await file_repo.add(
                            filepath, filename, total_pages, folder
                        )
                    else:
                        file = file_entry_exists

                    page = await page_repo.add(page_id, file)
                    embedding = await embedding_repo.add(vector_embedding, page)
                    await flattened_embedding_repo.add(vector_embedding, embedding)

        # Update the planner statistics to optimize query performance
        await execute_analyze()

        ctx = IndexingContext(IndexingStrategyFactory.create_strategy("HNSW"))
        await ctx.execute_indexing_strategy()

    async def search(
        self, query: str, top_k: int = 10
    ) -> Optional[List[Dict[str, Any]]]:
        """Search for the relevant file based on the query

        Args:
            query (str): Description about a file that user is searching
            top_k (int, optional): Top number of retrieved files. Defaults to 3.

        Returns:
            Optional[List[Dict[str, Any]]]: If the relevant file is found,
            it returns the page embeddings along with its metadata. Or None
        """

        # Run similarity search over the page embeddings for all the pages in the collection
        # top_indices has the shape of
        # tensor([[12,  0, 14],
        # [15, 14, 11]])
        qs = self.embed_query(query)

        # Based on https://github.com/pytorch/pytorch/issues/109873
        query_embeddings = [
            np.array(elem)
            for embed in qs
            for elem in embed.view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16)
        ]

        embeddings = await asyncio.to_thread(self.embed_images)

        session = async_session()
        await self.upsert_doc_embeddings(session=session, embeddings=embeddings)
        query_repo = QueryRepository(Query, session=session)
        await query_repo.add(query, query_embeddings, self.user_id)

        ctx = SearchContext(SearchStrategyFactory.create_strategy("ANNHNSWHamming"))
        return await ctx.execute_search_strategy(query_embeddings, top_k)
