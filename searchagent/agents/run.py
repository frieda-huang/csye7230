import asyncio

import fire
from searchagent.agents.api import Agent, create_agents
from searchagent.agents.common.multi_turn import execute_turns, prompt_to_turn
from searchagent.agents.factory import AgentFactory


async def run_main():

    factory = AgentFactory()

    await create_agents(factory)

    triange_agent = factory.get_agent(Agent.TRIAGE)

    await execute_turns(
        agent_config=triange_agent.agent_config,
        custom_tools=triange_agent.custom_tools,
        turn_inputs=[prompt_to_turn(content="Find me a paper on colpali")],
    )


def main():
    asyncio.run(run_main())


if __name__ == "__main__":
    fire.Fire(main)
