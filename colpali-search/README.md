# Colpali-Search

Colpali-Search is a microservice within the SearchAgent app. It functions as a PDF retrieval tool, enabling a single LLM agent to efficiently search and retrieve PDF files based on user queries. Under the hood, it leverages ColPali, a state-of-the-art retrieval framework that combines multi-vector retrieval techniques with vision-language models for highly efficient and accurate search capabilities.

## API Endpoints

| **Category**   | **Method** | **Endpoint**                     | **Description**                                                                                                                                                             |
| -------------- | ---------- | -------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Default**    | `GET`      | `/`                              | Fetches basic information about the microservice.                                                                                                                           |
|                | `POST`     | `/api/v1/search`                 | Executes a search query across indexed files using ColPali.                                                                                                                 |
|                | `POST`     | `/api/v1/benchmark`              | Benchmarks the search system with [vidore/syntheticDocQA_artificial_intelligence_test](https://huggingface.co/datasets/vidore/syntheticDocQA_artificial_intelligence_test). |
| **Embeddings** | `POST`     | `/api/v1/embeddings/file`        | Generates embeddings for a single file using the ColPali model.                                                                                                             |
|                | `POST`     | `/api/v1/embeddings/files`       | Generates embeddings for multiple files in batch mode.                                                                                                                      |
|                | `POST`     | `/api/v1/embeddings/benchmark`   | Generate embeddings for the vidore dataset to perform benchmarking.                                                                                                         |
| **Files**      | `GET`      | `/api/v1/files`                  | Retrieves a list of all indexed files.                                                                                                                                      |
|                | `GET`      | `/api/v1/files/{id}`             | Fetches details of a specific file by its ID.                                                                                                                               |
|                | `DELETE`   | `/api/v1/files/{id}`             | Deletes a file by its ID.                                                                                                                                                   |
| **Index**      | `GET`      | `/api/v1/index`                  | Lists all supported indexing strategies.                                                                                                                                    |
|                | `GET`      | `/api/v1/index/current-strategy` | Retrieves the currently active indexing strategy.                                                                                                                           |
|                | `POST`     | `/api/v1/index/reset-strategy`   | Resets the indexing strategy to the default configuration (i.e., HNSW with Cosine Similarity).                                                                              |
|                | `POST`     | `/api/v1/index/{strategy}`       | Configures a new indexing strategy (e.g., HNSW with Binary Quantization and Hamming Distance).                                                                              |

## Local setup

1.

2. Build the docker image
   `docker build -t colpali-search .`

3. Run the docker container
   `docker run -d -p 8000:8000 --name colpali-search colpali-search`

## Testing

We use Pytest and [pytest-asyncio](https://pytest-asyncio.readthedocs.io/en/latest/index.html#) for asynchronous testing. To run the tests, simply execute `pytest`
