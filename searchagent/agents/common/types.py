from enum import Enum
from typing import List

from llama_stack_client.types.agent_create_params import AgentConfig
from llama_stack_client.types.tool_param_definition_param import (
    ToolParamDefinitionParam,
)
from pydantic import BaseModel
from searchagent.agents.common.custom_tools import CustomTool


class AgentType(Enum):
    triage_agent = "triage_agent"
    file_retrieval_agent = "file_retrieval_agent"
    sync_agent = "sync_agent"
    index_agent = "index_agent"
    embed_agent = "embed_agent"

    def __str__(self):
        return self.value


class AgentWithCustomToolExecutor(BaseModel):
    name: str
    agent_id: str
    session_id: str
    agent_config: AgentConfig
    custom_tools: List[CustomTool]

    class Config:
        arbitrary_types_allowed = True

    @staticmethod
    def from_dict(data: dict) -> "AgentWithCustomToolExecutor":
        agent_config = AgentConfig(**data["agent_config"])

        return AgentWithCustomToolExecutor(
            name=data["name"],
            agent_id=data["agent_id"],
            session_id=data["session_id"],
            agent_config=agent_config,
            custom_tools=data["custom_tools"],
        )


# FIXME: This is a hack to bypass the JSON serialization error in custom tool's get_params_definition()
custom_tool_handoff_agent_params = {
    "query": ToolParamDefinitionParam(
        param_type="str",
        description="Determines which specialized agent or custom tool should handle the task",
        required=True,
    )
}
