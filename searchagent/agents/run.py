import asyncio

import fire
from llama_stack_client import LlamaStackClient
from searchagent.agents.api import Agent, AgentManager
from searchagent.agents.common.factory import AgentFactory
from searchagent.agents.common.multi_turn import execute_turns, prompt_to_turn


async def run_main():
    client = LlamaStackClient(base_url="http://localhost:11434")
    factory = AgentFactory(client)
    agent_manager = AgentManager(factory)

    await agent_manager.create_agents()

    triage_agent = factory.get_agent(Agent.TRIAGE)

    await execute_turns(
        triage_agent,
        turn_inputs=[prompt_to_turn(content="Find me a paper on colpali")],
    )


def main():
    asyncio.run(run_main())


if __name__ == "__main__":
    fire.Fire(main)
