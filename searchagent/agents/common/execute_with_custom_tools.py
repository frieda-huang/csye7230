# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


import json
from typing import AsyncGenerator, List, Optional, Union

from llama_stack_client.types import Attachment, UserMessage
from llama_stack_client.types.agent_create_params import AgentConfig
from loguru import logger
from searchagent.agents.common.custom_tools import (
    CustomTool,
    Message,
    ToolResponseMessage,
)


class AgentWithCustomToolExecutor:
    def __init__(
        self,
        name: str,
        agent_id: str,
        session_id: str,
        agent_config: AgentConfig,
        custom_tools: List[CustomTool],
    ):
        self.name = (name,)
        self.agent_id = agent_id
        self.session_id = session_id
        self.agent_config = agent_config
        self.custom_tools = custom_tools

    def to_dict(self):
        return {
            "name": self.name,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
            "agent_config": self.agent_config,
            "custom_tools": [ct.to_dict() for ct in self.custom_tools],
        }

    @staticmethod
    def from_dict(dict: dict):
        return AgentWithCustomToolExecutor(**dict)

    async def execute_turn(
        self,
        messages: List[Union[UserMessage, ToolResponseMessage]],
        attachments: Optional[List[Attachment]] = None,
        stream: bool = True,
    ) -> AsyncGenerator:
        from searchagent.agents.app_context import client
        from searchagent.agents.tools.routines import PDFSearchTool

        tools_dict = {t.get_name(): t for t in self.custom_tools}

        current_messages = messages.copy()

        while True:
            response = client.agents.turn.create(
                agent_id=self.agent_id,
                session_id=self.session_id,
                messages=current_messages,
                attachments=attachments,
                stream=True,
            )
            turn = None
            for chunk in response:
                if chunk.event.payload.event_type != "turn_complete":
                    yield chunk
                else:
                    turn = chunk.event.payload.turn

            message = turn.output_message

            if len(message.tool_calls) == 0:
                yield chunk
                return

            if message.stop_reason == "out_of_tokens":
                yield chunk
                return

            for tool_call in message.tool_calls:
                if tool_call.tool_name not in tools_dict:
                    m = ToolResponseMessage(
                        call_id=tool_call.call_id,
                        tool_name=tool_call.tool_name,
                        content=f"Unknown tool `{tool_call.tool_name}` was called. Try again with something else",
                        role="ipython",
                    )
                    next_message = m
                else:
                    tool = tools_dict[tool_call.tool_name]
                    result_messages = await execute_custom_tool(tool, message)

                    for result_message in result_messages:
                        custom_tool = json.loads(result_message.content)

                        if custom_tool["name"] == "pdf_search":
                            result = await PDFSearchTool.from_dict(
                                custom_tool, input_dir="."
                            ).run_impl("find the page about limitation of colpali")
                            logger.debug(result)

                        next_message = result_message

                        yield next_message
                        current_messages = [next_message]


async def execute_custom_tool(tool: CustomTool, message: Message) -> List[Message]:
    return await tool.run([message])
