from dataclasses import asdict
from typing import Dict

from llama_stack_client.types.tool_param_definition_param import (
    ToolParamDefinitionParam,
)
from searchagent.agents.common.custom_tools import SingleMessageCustomTool
from searchagent.agents.common.types import AgentType


class TransferToSyncAgent(SingleMessageCustomTool):
    def get_name(self) -> str:
        return "transfer_to_sync_agent"

    def get_description(self) -> str:
        return "Call this when file changes are detected"

    def get_params_definition(self) -> Dict[str, ToolParamDefinitionParam]:
        return {}

    async def run_impl(self):
        from searchagent.agents.app_context import factory

        return asdict(factory.get_agent(AgentType.sync_agent))
