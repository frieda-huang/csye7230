# Search2.0

## Process

1. We use LLama3.1 to generate a synthetic evaluation dataset.

2. We use the [CVPR 2019 Papers](https://www.kaggle.com/datasets/paultimothymooney/cvpr-2019-papers) dataset as our source of PDF documents.

3. We use [LlamaIndex](https://www.llamaindex.ai/) to create a baseline for our evaluation.

## Dataset

We use the [CVPR 2019 Papers](https://www.kaggle.com/datasets/paultimothymooney/cvpr-2019-papers) dataset from Kaggle. This dataset includes over 1,000 academic papers from the Computer Vision and Pattern Recognition (CVPR) 2019 conference

## Libraries

- [LlamaIndex](https://www.llamaindex.ai/): Data framework for LLM applications
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

## Resources

- [Project Proposal Part A](https://docs.google.com/document/d/1ojm1jtU8u-KRpF2hjG2bRb_PP1dPwSrfAUV27Sl0KeQ/edit?usp=sharing)
- [UML diagrams](https://drive.google.com/file/d/1AIpMmYtItZ8XGqRUUux1majA1Ue5sLSE/view?usp=sharing)

- [Python: Production-Level Coding Practices](https://medium.com/red-buffer/python-production-level-coding-practices-4c39246e0233)
- [RAG Evaluation](https://huggingface.co/learn/cookbook/rag_evaluation)
