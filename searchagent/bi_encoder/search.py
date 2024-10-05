from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class SemanticSearch:
    def __init__(self, documents):
        self.documents = documents
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.document_embeddings = self.model.encode(self.documents)

    def search(self, query):
        """Search for the top relevant documents based on the query."""
        # Embed the query
        query_embedding = self.model.encode([query])

        # Compute cosine similarity
        similarities = cosine_similarity(query_embedding, self.document_embeddings).flatten()

        # Get the top 3 results
        top_indices = similarities.argsort()[-3:][::-1]
        results = [{"document": self.documents[i], "similarity": similarities[i]} for i in top_indices]

        return results


if __name__ == "__main__":
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
