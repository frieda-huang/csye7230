from typing import Dict

from llama_stack_client.types.tool_param_definition_param import (
    ToolParamDefinitionParam,
)
from searchagent.agents.common.custom_tools import SingleMessageCustomTool
from searchagent.sync_manager.base import monitor


class FileMonitor(SingleMessageCustomTool):
    def __init__(self) -> None:
        self.monitor = monitor()

    def get_name(self) -> str:
        return "file_monitor"

    def get_description(self) -> str:
        return "Continuously monitor your files for any changes"

    def get_params_definition(self) -> Dict[str, ToolParamDefinitionParam]:
        return NotImplementedError

    async def run_impl(self):
        return monitor()
