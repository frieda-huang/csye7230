import json
from abc import abstractmethod
from typing import List

from llama_stack_client.types import CompletionMessage, ToolResponseMessage
from loguru import logger
from searchagent.agents.common.custom_tools import CustomTool
from searchagent.agents.common.execute_with_custom_tools import (
    AgentWithCustomToolExecutor,
)


class SingleMessageCustomTool(CustomTool):
    """
    Helper class to handle custom tools that take a single message
    Extending this class and implementing the `run_impl` method will
    allow for the tool be called by the model and the necessary plumbing.
    """

    async def run(self, messages: List[CompletionMessage]) -> List[ToolResponseMessage]:

        assert len(messages) == 1, "Expected single message"

        message = messages[0]

        tool_call = message.tool_calls[0]

        result = []

        try:
            response = await self.run_impl(**tool_call.arguments)
            custom_tools = AgentWithCustomToolExecutor(**response).custom_tools

            for custom_tool in custom_tools:
                response_str = json.dumps(custom_tool, ensure_ascii=False)

                message = ToolResponseMessage(
                    call_id=tool_call.call_id,
                    tool_name=tool_call.tool_name,
                    content=response_str,
                    role="ipython",
                )

                result.append(message)

        except Exception as e:
            logger.error(f"Error when running tool: {e}")

        return result

    @abstractmethod
    async def run_impl(self, *args, **kwargs):
        raise NotImplementedError()
