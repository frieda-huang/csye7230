from dataclasses import asdict
from typing import Dict

from llama_stack_client.types.tool_param_definition_param import (
    ToolParamDefinitionParam,
)
from searchagent.agents.common.custom_tools import SingleMessageCustomTool
from searchagent.agents.common.types import AgentType


class TransferToIndexAgent(SingleMessageCustomTool):
    def get_name(self) -> str:
        return "transfer_to_index_agent"

    def get_description(self) -> str:
        return "Call this when embeddings need to be indexed"

    def get_params_definition(self) -> Dict[str, ToolParamDefinitionParam]:
        return {}

    async def run_impl(self):
        from searchagent.agents.app_context import factory

        return asdict(factory.get_agent(AgentType.index_agent))
