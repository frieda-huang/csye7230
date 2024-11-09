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
from searchagent.colpali.schemas import ImageMetadata
from searchagent.colpali.search_engine.context import IndexingContext, SearchContext
from searchagent.colpali.search_engine.strategy_factory import (
    IndexingStrategyFactory,
    SearchStrategyFactory,
)
from searchagent.db_connection import async_session, get_table_names
from searchagent.models import Embedding, File, FlattenedEmbedding, Folder, Page, Query
from searchagent.ragmetrics.metrics import measure_latency_for_gpu, measure_vram
from searchagent.utils import VectorList, batch_processing, create_dummy_input, get_now
from sqlalchemy import select, text
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
        self.embeddings_by_page_id = {}
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

    @measure_latency_for_gpu(dummy_input_fn=lambda: create_dummy_input())
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

    @measure_vram()
    def embed_images(self) -> List[torch.Tensor]:
        # TODO: Set up a Cron job to sync files and their embeddings each night
        # TODO: Dynamically identify which PDF images need to be indexed
        """Embed images using custom Dataset
        Check this link on PyTorch custom Dataset:
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        """
        processed_images = PDFImagesDataset(self.images, self.pdf_metadata)
        return self._embed(processed_images, self.image_collate_fn)

    @measure_vram()
    def embed_query(self, query: str) -> List[torch.Tensor]:
        return self._embed([query], self.processor.process_queries)

    def build_embed_metadata(self, embeddings: List[torch.Tensor], mdata: List[Dict]):
        """Structure embeddings for faster retrieval
        {
            "page_1": {
                "embedding": [0.1, 0.2, 0.3, ...],
                "metadata": {
                    "pdf_id": "pdf_1",
                    "page_id": "page_1",
                    "total_pages": 10,
                    "filename": "document.pdf",
                    "file_path": "/path/to/document.pdf",
                },
                "created_at": "2024-09-30T12:00:00Z",
                "modified_at": "2024-09-30T12:00:00Z",
            },
            ...
        }
        """
        # Move to CPU to reflect correct vram measurement in embed_images
        embeddings = [e.to("cpu") for e in embeddings]

        chunk_size = 1000
        embed_len = len(embeddings)
        for i in range(0, embed_len, chunk_size):
            chunk_embeddings = embeddings[i : i + chunk_size]
            chunk_metadata = mdata[i : i + chunk_size]

            update_dict = {
                f'{metadata["page_id"]}_{metadata["pdf_id"]}': {
                    "embedding": embedding.tolist(),
                    "metadata": metadata,
                    "created_at": get_now(),
                    "modified_at": get_now(),
                }
                for embedding, metadata in zip(chunk_embeddings, chunk_metadata)
            }
            self.embeddings_by_page_id.update(update_dict)

    async def upsert_query_embeddings(self, query: str, query_embeddings: VectorList):
        async with async_session.begin() as session:
            q = Query(
                text=query,
                vector_embedding=query_embeddings,
                created_at=get_now(),
                user_id=self.user_id,
            )
            session.add(q)

    async def does_folder_entry_exist(self):
        folder_path = str(self.input_dir)

        async with async_session.begin() as session:
            folder_stm = select(Folder).filter_by(folder_path=folder_path)
            return await session.scalar(folder_stm)

    def get_scores(self, qs: List[torch.Tensor], embeddings: List[torch.Tensor]):
        return self.processor.score(qs, embeddings)

    async def upsert_doc_embeddings(self):
        """Upsert embeddings to PostgreSQL"""

        async with async_session.begin() as session:

            async def add_embeddings_to_session(batch):

                for key, value in batch:
                    parts = key.split("_")
                    page_id = parts[0]
                    embeddings = value["embedding"]
                    vector_embedding = [np.array(e) for e in embeddings]

                    metadata = value["metadata"]
                    filename = metadata["filename"]
                    filepath = metadata["filepath"]
                    total_pages = metadata["total_pages"]
                    folder_name, folder_path = "", ""

                    if not self.benchmark:
                        folder_name = str(self.input_dir.name)
                        folder_path = str(self.input_dir)

                    # Check if folder already exists
                    folder_entry_exists = (
                        await self.does_folder_entry_exist()
                        if not self.benchmark
                        else False
                    )
                    if not folder_entry_exists:
                        folder = Folder(
                            folder_name=folder_name,
                            folder_path=folder_path,
                            created_at=get_now(),
                            user_id=self.user_id,
                        )
                        session.add(folder)

                    # Check if file already exists
                    file_stm = select(File).filter_by(
                        filepath=filepath, filename=filename
                    )
                    file = await session.scalar(file_stm)
                    if not file:
                        file = File(
                            filename=filename,
                            filepath=filepath,
                            filetype="pdf",
                            total_pages=total_pages,
                            last_modified=get_now(),
                            created_at=get_now(),
                            folder=folder,
                        )
                        session.add(file)

                    page = Page(
                        page_number=page_id,
                        last_modified=get_now(),
                        created_at=get_now(),
                        file=file,
                    )
                    session.add(page)

                    embedding = Embedding(
                        vector_embedding=vector_embedding,
                        page=page,
                        last_modified=get_now(),
                        created_at=get_now(),
                    )
                    session.add(embedding)

                    async def add_flattened_embeddings_to_session(batch: List[Any]):
                        for e in batch:
                            flattened_embedding = FlattenedEmbedding(
                                vector_embedding=e,
                                last_modified=get_now(),
                                created_at=get_now(),
                                embedding=embedding,
                            )
                            session.add(flattened_embedding)

                    await batch_processing(
                        original_list=vector_embedding,
                        batch_size=300,
                        func=add_flattened_embeddings_to_session,
                    )

            await batch_processing(
                original_list=self.embeddings_by_page_id.items(),
                batch_size=100,
                func=add_embeddings_to_session,
            )

            # Update the planner statistics to optimize query performance
            table_names = await get_table_names()
            for table_name in table_names:
                if table_name == "user":
                    continue
                sql = text(f"ANALYZE {table_name};")
                await session.execute(sql)

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

        await asyncio.to_thread(self.build_embed_metadata, embeddings, self.metadata)

        await self.upsert_doc_embeddings()

        await self.upsert_query_embeddings(query, query_embeddings)
        ctx = SearchContext(SearchStrategyFactory.create_strategy("ANNHNSWHamming"))
        return await ctx.execute_search_strategy(query_embeddings, top_k)
