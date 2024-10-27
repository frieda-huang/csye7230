from enum import Enum
from typing import Callable, List, Union

from llama_stack_client.types.agent_create_params import AgentConfig
from pydantic import BaseModel
from searchagent.agents.common.custom_tools import CustomTool

AgentFunction = Callable[[], Union[str, "Agent", dict]]


class Agent(BaseModel):
    name: str
    agent_config: AgentConfig
    custom_tools: List[CustomTool]
    functions: List[AgentFunction] = []

    class Config:
        arbitrary_types_allowed = True


class AgentType(Enum):
    triage_agent = "triage_agent"
    file_retrieval_agent = "file_retrieval_agent"
    sync_agent = "sync_agent"
    index_agent = "index_agent"
    embed_agent = "embed_agent"

    def __str__(self):
        return self.value
