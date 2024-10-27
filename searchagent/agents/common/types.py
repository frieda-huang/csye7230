from enum import Enum


class AgentType(Enum):
    triage_agent = "triage_agent"
    file_retrieval_agent = "file_retrieval_agent"
    sync_agent = "sync_agent"
    index_agent = "index_agent"
    embed_agent = "embed_agent"

    def __str__(self):
        return self.value
