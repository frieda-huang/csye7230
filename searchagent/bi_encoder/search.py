from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Optional, Dict
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

