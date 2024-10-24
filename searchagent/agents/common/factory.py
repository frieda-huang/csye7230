from typing import List, Optional

from searchagent.agents.common.client_utils import (
    CustomTool,
    QuickToolConfig,
    make_agent_config_with_custom_tools,
)
from searchagent.agents.common.types import Agent, AgentFunction


class AgentFactory:
    def __init__(self):
        self._agents = {}

    @staticmethod
    async def create_agent(
        name: str,
        functions: List[AgentFunction],
        custom_tools: List[CustomTool] = [],
    ) -> Agent:
        agent_config = await make_agent_config_with_custom_tools(
            tool_config=QuickToolConfig(
                custom_tools=custom_tools,
                prompt_format="function_tag",
            )
        )
        return Agent(
            name=name,
            agent_config=agent_config,
            custom_tools=custom_tools,
            functions=functions,
        )

    async def create_and_register_agent(
        self,
        name: str,
        functions: List[AgentFunction],
        custom_tools: List[CustomTool] = [],
    ):
        agent = await self.create_agent(name, functions, custom_tools)
        self._agents[name] = agent

    def get_agent(self, name: str) -> Optional[Agent]:
        return self._agents.get(name)
