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
