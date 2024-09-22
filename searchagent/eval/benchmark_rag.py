# Based on https://huggingface.co/learn/cookbook/rag_evaluation

import json
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import datasets
import ollama
import pandas as pd
from datasets import Dataset
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStore
from searchagent.eval.prompts.qa_generation_prompt import QA_generation_prompt
from searchagent.eval.prompts.question_critique_prompt import (
    question_groundedness_critique_prompt,
    question_relevance_critique_prompt,
    question_standalone_critique_prompt,
)
from searchagent.utils import project_paths
from tqdm.auto import tqdm


class QAGenerationEvaluator:
    def __init__(
        self,
        model="llama3.1:8b-instruct-fp16",
        n_generations=2000,
        dataset_name="m-ric/huggingface_doc",
    ):

        self.model = model
        self.n_generations = n_generations
        self.dataset_name = dataset_name
        self.ds = datasets.load_dataset(dataset_name, split="train")
        self.langchain_docs = [
            LangchainDocument(
                page_content=doc["text"], metadata={"source": doc["source"]}
            )
            for doc in tqdm(self.ds)
        ]

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            add_start_index=True,
            separators=["\n\n", "\n", ".", " ", ""],
        )

        self.docs_processed = self._process_documents()
        self.outputs = None
        self.generated_questions = None

    def _process_documents(self) -> List:
        docs_processed = []
        for doc in self.langchain_docs:
            docs_processed += self.text_splitter.split_documents([doc])
        return docs_processed

    def call_llm(self, prompt: str) -> str:
        res = ollama.generate(
            model=self.model,
            prompt=prompt,
            options={"num_predict": 1000},
        )
        return res["response"]

    def generate_qa_pairs(self) -> List[Dict]:
        outputs = []
        for sampled_context in tqdm(
            random.sample(self.docs_processed, self.n_generations)
        ):
            # Generate QA couple
            output_QA_couple = self.call_llm(
                QA_generation_prompt.format(context=sampled_context.page_content)
            )
            try:
                question = output_QA_couple.split("Factoid question: ")[-1].split(
                    "Answer: "
                )[0]
                answer = output_QA_couple.split("Answer: ")[-1]
                assert len(answer) < 300, "Answer is too long"
                outputs.append(
                    {
                        "context": sampled_context.page_content,
                        "question": question,
                        "answer": answer,
                        "source_doc": sampled_context.metadata["source"],
                    }
                )
            except Exception:
                continue

        return outputs

    def critique_qa_pairs(self):
        self.outputs = self.generate_qa_pairs()
        for output in tqdm(self.outputs):
            evaluations = {
                "groundedness": self.call_llm(
                    question_groundedness_critique_prompt.format(
                        context=output["context"], question=output["question"]
                    ),
                ),
                "relevance": self.call_llm(
                    question_relevance_critique_prompt.format(
                        question=output["question"]
                    ),
                ),
                "standalone": self.call_llm(
                    question_standalone_critique_prompt.format(
                        question=output["question"]
                    ),
                ),
            }
            try:
                for criterion, evaluation in evaluations.items():
                    score, eval = (
                        int(evaluation.split("Total rating: ")[-1].strip()),
                        evaluation.split("Total rating: ")[-2].split("Evaluation: ")[1],
                    )
                    output.update(
                        {
                            f"{criterion}_score": score,
                            f"{criterion}_eval": eval,
                        }
                    )
            except Exception:
                continue
        return self

    def filter_qa_pairs(self):
        generated_questions = pd.DataFrame.from_dict(self.outputs)
        self.generated_questions = generated_questions.loc[
            (generated_questions["groundedness_score"] >= 4)
            & (generated_questions["relevance_score"] >= 4)
            & (generated_questions["standalone_score"] >= 4)
        ]
        return self

    def create_eval_dataset(self) -> Dataset:
        return datasets.Dataset.from_pandas(
            self.generated_questions, split="train", preserve_index=False
        )

    def evaluate_and_filter_qa_pairs(self):
        return self.critique_qa_pairs().filter_qa_pairs().create_eval_dataset()

    def save_synthetic_test_data(self):
        output_filename = (
            f"{self.dataset_name.replace('/', '~')}_{len(self.generated_questions)}.csv"
        )
        df = self.evaluate_and_filter_qa_pairs()
        OUTPUT_TESTDATA_DIR = project_paths.DATA / "synthetic_testdata"

        if not os.path.exists(OUTPUT_TESTDATA_DIR):
            os.mkdir(OUTPUT_TESTDATA_DIR)

        df.to_csv(OUTPUT_TESTDATA_DIR / output_filename)

    @staticmethod
    def run_rag_tests(
        answer_with_rag: Callable[
            [str, VectorStore, Optional[Any]],
            Tuple[str, List[LangchainDocument]],
        ],
        eval_dataset: datasets.Dataset,
        knowledge_index: VectorStore,
        output_file: str,
        reranker: Optional[Any] = None,
        verbose: Optional[bool] = True,
        test_settings: Optional[str] = None,  # To document the test settings used
    ):
        """Runs RAG tests on the given dataset and saves the results to the given output file."""
        try:  # load previous generations if they exist
            with open(output_file, "r") as f:
                outputs = json.load(f)
        except Exception:
            outputs = []

        for example in tqdm(eval_dataset):
            question = example["question"]
            if question in [output["question"] for output in outputs]:
                continue

            answer, relevant_docs = answer_with_rag(
                question, knowledge_index, reranker=reranker
            )
            if verbose:
                print("=======================================================")
                print(f"Question: {question}")
                print(f"Answer: {answer}")
                print(f'True answer: {example["answer"]}')
            result = {
                "question": question,
                "true_answer": example["answer"],
                "source_doc": example["source_doc"],
                "generated_answer": answer,
                "retrieved_docs": [doc for doc in relevant_docs],
            }
            if test_settings:
                result["test_settings"] = test_settings
            outputs.append(result)

            with open(output_file, "w") as f:
                json.dump(outputs, f)


# Usage
e = QAGenerationEvaluator()
e.save_synthetic_test_data()
