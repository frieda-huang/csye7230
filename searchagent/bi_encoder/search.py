from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional, Dict
from searchagent.file_system.base import FileSystemManager
import os
import numpy as np



class SemanticSearch:
    def __init__(self, documents: List[str]):
        self.documents = documents
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.document_embeddings = self.model.encode(self.documents)

    def search(self, query: str) -> List[Dict[str, float]]:
        """Search for the top relevant documents based on the query."""
        # Embed the query
        query_embedding = self.model.encode([query])

        # Compute cosine similarity
        similarities = cosine_similarity(query_embedding, self.document_embeddings).flatten()

        # Get the top 3 results
        top_indices = similarities.argsort()[-3:][::-1]
        results = [{"document": self.documents[i], "similarity": similarities[i]} for i in top_indices]

        return results


def run_test_semantic_search():
    # Sample documents
    documents = [
        "The cat sits on the mat.",
        "Dogs are great companions.",
        "I love to play football.",
        "Cats are independent animals.",
        "Football is played worldwide.",
        "Pets can be very loyal.",
        "The quick brown fox jumps over the lazy dog."
    ]

    # Create a SemanticSearch object
    semantic_search = SemanticSearch(documents)

    # Get user input for the query
    query = input("Enter your search query: ")

    # Perform the search
    results = semantic_search.search(query)

    # Display the results
    print("Top matching documents:")
    for result in results:
        print(f"Document: {result['document']}, Similarity: {result['similarity']:.4f}")



def run_semantic_search_file_names():
    # Get the current project home directory
    directory = os.getcwd()  # Current working directory

    # Create a FileSystemManager object
    file_manager = FileSystemManager(dir=directory)

    # Get list of document file names
    documents = file_manager.list_files(file_manager.retrievable_files)

    # Create a SemanticSearch object
    semantic_search = SemanticSearch(documents)

    # Get user input for the query
    query = input("Enter your search query: ")

    # Perform the search
    results = semantic_search.search(query)

    # Display the results
    print("Top matching documents:")
    for result in results:
        print(f"Document: {result['document']}, Similarity: {result['similarity']:.4f}")



