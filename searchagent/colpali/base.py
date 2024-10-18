import json
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
from searchagent.colpali.models import ImageMetadata, StoredImageData
from searchagent.colpali.pdf_images_dataset import PDFImagesDataset
from searchagent.colpali.pdf_images_processor import PDFImagesProcessor
from searchagent.colpali.search_engine.strategy_factory import SearchStrategyFactory
from searchagent.db_connection import Session
from searchagent.models import (
    Embedding,
    File,
    FlattenedEmbedding,
    Folder,
    Page,
    Query,
)
from searchagent.utils import VectorList, get_now
from sqlalchemy import select
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BatchFeature, PreTrainedModel


class ColPaliRag:
    def __init__(
        self,
        input_dir: Union[Path, str],
        model_name: Optional[str] = None,
        store_locally: bool = True,
        hf_api_key: Optional[str] = None,
    ):
        """
        A RAG system using a visual language model called Colpali to process PDF files

        Args:
            query (str): search query
            input_dir (Union[Path, str]): Path to the PDF directory
            model_name (Optional[str]): Name of a Colpali pretrained model
                Default is "vidore/colpali-v1.2"
            store_locally (Optional[bool]): Whether to store index locally or upload to the VectorDB
                Default is True
        """

        # Lazy loadings
        self._model = None
        self._processor: Optional[ColPaliProcessor] = None
        self.embeddings_by_page_id = {}
        self.stored_embeddings: Dict[str, StoredImageData] = {}
        self.stored_filepath = None
        self.user_id = 1  # Replace with actual user ID
        self.metadata = None

        self.input_dir = Path(input_dir)
        self.store_locally = store_locally
        self.device = get_torch_device()
        self.model_name = model_name or "vidore/colpali-v1.2"
        self.hf_api_key = hf_api_key or os.getenv("HF_API_KEY")

        self.pdf_processor = PDFImagesProcessor.convert_pdf2image_from_dir(
            str(self.input_dir)
        )
        self.images = self.pdf_processor.images_list
        self.pdf_metadata = self.pdf_processor.pdf_metadata

        if not self.hf_api_key:
            raise ValueError("HuggingFace API key is required")

        if not self.input_dir.is_dir():
            raise ValueError(f"Input directory does not exist: {input_dir}")

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
                mdata.extend(meta.to_json() for meta in metadata)
                self.metadata = mdata
            else:
                batches = {k: v.to(self.device) for k, v in batch.items()}

            # Move images to GPU and get embeddings
            batch_embeddings = self.model(**batches)

            # Embeddings for a single PDF page; it is of the shape (1030, 128)
            # 32x32=1024 image patches and 6 instruction text tokens
            individual_embeddings = list(torch.unbind(batch_embeddings.to("cpu")))

            # Collect all embeddings
            embeddings.extend(individual_embeddings)

        return embeddings

    def embed_images(self) -> List[torch.Tensor]:
        """Embed images using custom Dataset
        Check this link on PyTorch custom Dataset:
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        """
        processed_images = PDFImagesDataset(self.images, self.pdf_metadata)
        embeddings = self._embed(processed_images, self.image_collate_fn)

        self.build_embed_metadata(embeddings, self.metadata)

        if self.store_locally:
            self._store_index_locally()
        else:
            self.upsert_doc_embeddings()

        return embeddings

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
        self.embeddings_by_page_id.update(
            {
                f'{metadata["page_id"]}_{metadata["pdf_id"]}': {
                    "embedding": embedding.tolist(),
                    "metadata": metadata,
                    "created_at": get_now(),
                    "modified_at": get_now(),
                }
                for embedding, metadata in zip(embeddings, mdata)
            }
        )

    def _store_index_locally(self):
        import os

        output_file = "embeddings_metadata.json"
        self.stored_filepath = f"{os.getcwd()}/{output_file}"
        with open(output_file, "w") as f:
            json.dump(self.embeddings_by_page_id, f, indent=4)

    def upsert_query_embeddings(self, query: str, query_embeddings: VectorList):
        with Session.begin() as session:
            q = Query(
                text=query,
                vector_embedding=query_embeddings,
                created_at=get_now(),
                user_id=self.user_id,
            )
            session.add(q)

    def upsert_doc_embeddings(self):
        """Upsert embeddings to PostgreSQL"""
        # TODO: It's getting a bit crazy, will refactor later

        for key, value in self.embeddings_by_page_id.items():
            parts = key.split("_")
            page_id = parts[0]
            embeddings = value["embedding"]
            vector_embedding = [np.array(e) for e in embeddings]

            metadata = value["metadata"]
            filename = metadata["filename"]
            filepath = metadata["filepath"]
            total_pages = metadata["total_pages"]
            folder_name = str(self.input_dir.name)
            folder_path = str(self.input_dir)

            with Session.begin() as session:
                # Check if folder already exists
                folder_stm = select(Folder).filter_by(folder_path=folder_path)
                folder = session.scalar(folder_stm)
                if not folder:
                    folder = Folder(
                        folder_name=folder_name,
                        folder_path=folder_path,
                        created_at=get_now(),
                        user_id=self.user_id,
                    )
                    session.add(folder)

                # Check if file already exists
                file_stm = select(File).filter_by(filepath=filepath)
                file = session.scalar(file_stm)
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

                for e in vector_embedding:
                    flattened_embedding = FlattenedEmbedding(
                        vector_embedding=e,
                        last_modified=get_now(),
                        created_at=get_now(),
                        embedding=embedding,
                    )
                    session.add(flattened_embedding)

    def load_stored_embeddings(self, filepath: str) -> Dict[str, StoredImageData]:
        """Load stored embeddings in memory"""
        with open(filepath, "r") as f:
            json_data = json.load(f)
            for page_pdf_id, data in json_data.items():
                self.stored_embeddings[page_pdf_id] = StoredImageData.from_json(data)
        return self.stored_embeddings

    def search(
        self, query: str, top_k: int = 3, filepath: Optional[str] = None
    ) -> Optional[List[Dict[str, Any]]]:
        """Search for the relevant file based on the query

        Args:
            query (str): Description about a file that user is searching
            top_k (int, optional): Top number of retrieved files. Defaults to 3.
            filepath (Optional[str], optional):
                Path to "embeddings_metadata.json"

        Returns:
            List[Dict[str, Any]]: If the relevant file is found,
            it returns the page embeddings along with its metadata
        """

        embeddings = (
            [data.embedding for data in self.load_stored_embeddings(filepath).values()]
            if filepath
            else self.embed_images()
        )

        if self.stored_filepath and not filepath:
            self.load_stored_embeddings(self.stored_filepath)

        # Run similarity search over the page embeddings for all the pages in the collection
        # top_indices has the shape of
        # tensor([[12,  0, 14],
        # [15, 14, 11]])
        qs = self.embed_query(query)
        scores = self.processor.score(qs, embeddings)
        _, top_indices = torch.topk(scores, k=top_k, dim=1)

        # Ensure even if top_k=1, top_indices will be a 1D tensor, preventing the iteration error
        if top_k == 1:
            top_indices = top_indices.squeeze().unsqueeze(0)
        else:
            top_indices = top_indices.squeeze()

        # Based on https://github.com/pytorch/pytorch/issues/109873
        query_embeddings = [
            np.array(elem)
            for embed in qs
            for elem in embed.view(dtype=torch.uint16).numpy().view(ml_dtypes.bfloat16)
        ]

        if self.stored_embeddings:
            # Create a mapping of (page_id, pdf_id) for retrieval
            indexed_metadata = [
                (data.metadata.page_id, data.metadata.pdf_id)
                for data in self.stored_embeddings.values()
            ]

            # This will return a list of (pdf_id, page_id) for top results
            top_metadata = [indexed_metadata[idx.item()] for idx in top_indices]
            return self.retrieve_page_info(top_metadata)

        self.upsert_query_embeddings(query, query_embeddings)

        ss = SearchStrategyFactory.create_search_strategy("ANNHNSWHamming")
        ss.search(query_embeddings, top_k)

    def retrieve_page_info(
        self, top_metadata: List[Tuple[str, int]]
    ) -> List[Dict[str, Any]]:
        if self.stored_embeddings:
            return [
                self.stored_embeddings[f"{page_id}_{pdf_id}"].to_json()
                for page_id, pdf_id in top_metadata
            ]
        return []
