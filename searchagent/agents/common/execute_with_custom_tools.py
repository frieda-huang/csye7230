# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# the root directory of this source tree.


from typing import AsyncGenerator, List, Optional, Union

from llama_stack_client.types import Attachment, ToolResponseMessage, UserMessage
from loguru import logger
from searchagent.agents.app_context import client
from searchagent.agents.common.custom_tools import CustomTool, Message
from searchagent.agents.common.types import AgentWithCustomToolExecutor


async def execute_turn(
    agent: AgentWithCustomToolExecutor,
    messages: List[Union[UserMessage, ToolResponseMessage]],
    attachments: Optional[List[Attachment]] = None,
    stream: bool = True,
) -> AsyncGenerator:
    # from searchagent.agents.app_context import client

    tools_dict = {t.get_name(): t for t in agent.custom_tools}

    current_messages = messages.copy()

    while True:
        response = client.agents.turn.create(
            agent_id=agent.agent_id,
            session_id=agent.session_id,
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

                if type(result_messages) is AgentWithCustomToolExecutor:
                    current_agent = result_messages
                    logger.debug(current_agent)

                next_message = result_messages[0]

            yield next_message
            current_messages = [next_message]


async def execute_custom_tool(
    tool: CustomTool, message: Message
) -> Union[AgentWithCustomToolExecutor, ToolResponseMessage]:
    return await tool.run([message])
