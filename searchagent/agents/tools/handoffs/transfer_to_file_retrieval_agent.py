from dataclasses import asdict
from typing import Dict

from llama_stack_client.types.tool_param_definition_param import (
    ToolParamDefinitionParam,
)
from searchagent.agents.app_context import factory
from searchagent.agents.common.custom_tools import SingleMessageCustomTool
from searchagent.agents.common.types import AgentType


class TransferToFileRetrievalAgent(SingleMessageCustomTool):
    def get_name(self) -> str:
        return "transfer_to_file_retrieval_agent"

    def get_description(self) -> str:
        return "Use this to handle user file search requests"

    def get_params_definition(self) -> Dict[str, ToolParamDefinitionParam]:
        return {}

    async def run_impl(self):
        return asdict(factory.get_agent(AgentType.file_retrieval_agent))
