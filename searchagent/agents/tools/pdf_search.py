from pathlib import Path
from typing import Dict, Union

from llama_stack_client.types.tool_param_definition_param import (
    ToolParamDefinitionParam,
)
from searchagent.agents.common.custom_tools import SingleMessageCustomTool
from searchagent.colpali.base import ColPaliRag


class PDFSearchTool(SingleMessageCustomTool):
    def __init__(self, input_dir: Union[Path, str]):
        self.colpali = ColPaliRag(input_dir, store_locally=False)

    def get_name(self) -> str:
        return "pdf_search"

    def get_description(self) -> str:
        return "Search PDF files for a given query"

    def get_params_definition(self) -> Dict[str, ToolParamDefinitionParam]:
        return {
            "query": ToolParamDefinitionParam(
                param_type="str",
                description="The search query used to locate a file",
                required=True,
            )
        }

    async def run_impl(self, query: str):
        return await self.colpali.search(query)
