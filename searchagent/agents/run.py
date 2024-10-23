import asyncio

import fire
from searchagent.agents.api import triage_agent
from searchagent.agents.common.multi_turn import execute_turns, prompt_to_turn


async def run_main():
    await execute_turns(
        agent_config=triage_agent.agent_config,
        custom_tools=[],
        turn_inputs=[prompt_to_turn(content="Find me a paper on colpali")],
    )


def main():
    asyncio.run(run_main())


if __name__ == "__main__":
    fire.Fire(main)
