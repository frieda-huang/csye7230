from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
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

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
document_embeddings = model.encode(documents)


def search(query):
    # Embed the query
    query_embedding = model.encode([query])

    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding, document_embeddings).flatten()

    # Get the top 3 results
    top_indices = similarities.argsort()[-3:][::-1]
    results = [{"document": documents[i], "similarity": similarities[i]} for i in top_indices]

    return results


