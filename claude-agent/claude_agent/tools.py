import json
from enum import Enum
from typing import List

import requests

BASE_URL = "http://54.188.147.87:8000/api/v1"


class ToolNames(str, Enum):
    colpali_search = "colpali_search"
    colpali_embed = "colpali_embed"


def colpali_search(query: str, top_k: int, email: str) -> dict:
    try:
        json_body = {"query": query, "top_k": top_k, "email": email}
        response = requests.post(f"{BASE_URL}/search", json=json_body)
        result = response.json()
        return json.dumps(result)
    except Exception as e:
        return f"Error: {e}"


def colpali_embed(filepaths: List[str]):
    try:
        files = [
            ("files", (filepath, open(filepath, "rb"), "application/pdf"))
            for filepath in filepaths
        ]
        response = requests.post(f"{BASE_URL}/embeddings/files", files=files)
        result = response.json()
        return json.dumps(result["metadata"])
    except Exception as e:
        return f"Error: {e}"


tools = [
    {
        "name": "colpali_search",
        "description": "Performs a search based on a query and additional parameters",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query must not be empty",
                },
                "top_k": {
                    "type": "integer",
                    "description": "top_k must be greater than 0",
                },
                "email": {
                    "type": "string",
                    "description": "The email address of the user making the request",
                },
            },
            "required": ["query", "top_k", "email"],
        },
    },
    {
        "name": "colpali_embed",
        "description": "Generates embeddings for PDF files",
        "input_schema": {
            "type": "object",
            "properties": {
                "filepaths": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "A list of file paths to the PDF files",
                }
            },
            "required": ["filepaths"],
        },
        "cache_control": {"type": "ephemeral"},
    },
]
