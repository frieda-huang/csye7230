# Search2.0

## Process

1. We use LLama3.1 to generate a synthetic evaluation dataset.

2. We use the [CVPR 2019 Papers](https://www.kaggle.com/datasets/paultimothymooney/cvpr-2019-papers) dataset as our source of PDF documents.

3. We use [LlamaIndex](https://www.llamaindex.ai/) to create a baseline for our evaluation.

## Dataset

We use the [CVPR 2019 Papers](https://www.kaggle.com/datasets/paultimothymooney/cvpr-2019-papers) dataset from Kaggle, containing over 1,000 academic papers from the CVPR 2019 conference. From this dataset, 5 papers were randomly selected to generate 10 test sets (`cvpr2019_5papers_testset_12q.csv`) using the [Ragas](https://docs.ragas.io/en/latest/index.html) framework. The dataset can be found at [friedahuang/cvpr2019_5papers_testset_12q](https://huggingface.co/datasets/friedahuang/cvpr2019_5papers_testset_12q) See `ragas_evaluate.py` for the implementation.

1.  Aafaq_Spatio-Temporal_Dynamics_and_Semantic_Attribute_Enriched_Visual_Encoding_for_Video_CVPR_2019_paper.pdf
2.  Aakur_A_Perceptual_Prediction_Framework_for_Self_Supervised_Event_Segmentation_CVPR_2019_paper.pdf
3.  Abati_Latent_Space_Autoregression_for_Novelty_Detection_CVPR_2019_paper.pdf
4.  Abavisani_Improving_the_Performance_of_Unimodal_Dynamic_Hand-Gesture_Recognition_With_Multimodal_CVPR_2019_paper.pdf
5.  Abbasnejad_A_Generative_Adversarial_Density_Estimator_CVPR_2019_paper.pdf

Additionally, we use the Huggingface dataset [(m-ric/huggingface_doc)](https://huggingface.co/datasets/m-ric/huggingface_doc) to generate 347 question-answer pairs (QA couples). The synthetically generated QA couples can be found at [friedahuang/m-ric_huggingface_doc_347](https://huggingface.co/datasets/friedahuang/m-ric_huggingface_doc_347). We will focus on this evaluation dataset because it has a larger volume and we've already established a baseline RAG system benchmarked against it. See `benchmark_rag.py` for the implementation.

## Libraries

- [LangChain](https://www.langchain.com/): Framework for LLM applications
- [Loguru](https://github.com/Delgan/loguru): Simplified Python logging
- [pre-commit](https://pre-commit.com/): Multi-language pre-commit hooks manager
- [Ruff](https://docs.astral.sh/ruff/): Fast Python linter and formatter

## Setup

Instructions on how to set up the project locally. For example:

1. Clone the repository:

   ```
   git clone https://github.com/frieda-huang/csye7230.git
   ```

2. Install dependencies:

   ```
   poetry install
   ```

3. Set up pre-commit hooks:
   ```
   pre-commit install
   ```

## System Overview

We will pre-index the user's home directory (also fully indexed by macOS Spotlight), which contains most user-accessible files and data. Home directory includes:

- Desktop
- Documents
- Downloads
- Pictures

## Example

```
>> response = searchagent.query("find csye7230 project proposal")
>> response.documents

Output:

Document(metadata={'source': '../proposals/csye7230_project_proposal_part_a.pdf'}, page_content='...')
Document(metadata={'source': '../proposals/csye7230_project_proposal_part_b.pdf'}, page_content='...')
Document(metadata={'source': '../proposals/csye7230_project_benchmarking_report.pdf'}, page_content='...')

>> response.answer

Output:

The following files match your query:

1. csye7230_project_proposal_part_a.pdf
`../proposals/csye7230_project_proposal_part_a.pdf`

This PDF contains the project proposal for CSYE7230, detailing objectives, methodologies, and expected outcomes.

2. csye7230_project_proposal_part_b.pdf
`../proposals/csye7230_project_proposal_part_b.pdf`

This PDF outlines the implementation plan for project proposal part B, focusing on architecture and design choices.

3. csye7230_project_benchmarking_report.pdf
`../proposals/csye7230_project_benchmarking_report.pdf`

This PDF presents the benchmarking report for CSYE7230, evaluating the performance metrics and analysis of the project components.
```

## Resources

- [Project Proposal Part A](https://docs.google.com/document/d/1ojm1jtU8u-KRpF2hjG2bRb_PP1dPwSrfAUV27Sl0KeQ/edit?usp=sharing)
- [UML diagrams](https://drive.google.com/file/d/1AIpMmYtItZ8XGqRUUux1majA1Ue5sLSE/view?usp=sharing)

- [Python: Production-Level Coding Practices](https://medium.com/red-buffer/python-production-level-coding-practices-4c39246e0233)
- [RAG Evaluation (LLM-as-a-judge)](https://huggingface.co/learn/cookbook/rag_evaluation)
- [Analyze file system and folder structures with Python](https://janakiev.com/blog/python-filesystem-analysis/)
- [Pytest Best Practices](https://realpython.com/pytest-python-testing/)
- [Implement semantic cache to improve a RAG system](https://huggingface.co/learn/cookbook/semantic_cache_chroma_vector_database)
- [Set up eval pipeline](https://www.youtube.com/watch?v=eLXF0VojuSs&t=140s)
- [Reranking](https://medium.com/google-cloud/reranking-3b5f351cb398)
- [Evaluating Chunking Strategies for Retrieval](https://research.trychroma.com/evaluating-chunking)
- [A Reddit post on a new chunking algorithm](https://www.reddit.com/r/LangChain/comments/1flhtxi/a_new_chunking_algorithm_proposal_semantically/)
- [5 levels of text splitting](https://github.com/FullStackRetrieval-com/RetrievalTutorials/blob/main/tutorials/LevelsOfTextSplitting/5_Levels_Of_Text_Splitting.ipynb)
- [A blog on ripgrep—a line-oriented search tool that recursively searches the current directory for a regex pattern](https://blog.burntsushi.net/ripgrep/)
- [microsearch—a search engine in 80 lines of Python](https://www.alexmolas.com/2024/02/05/a-search-engine-in-80-lines.html)
- [Agent architectures](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/)
- [Embedding Quantization](https://huggingface.co/blog/embedding-quantization)
- [Llama3.2 is here](https://ai.meta.com/blog/llama-3-2-connect-2024-vision-edge-mobile-devices/)