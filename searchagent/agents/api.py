from typing import Callable, List, Union

from pydantic import BaseModel
from searchagent.agents.common.client_utils import (
    QuickToolConfig,
    make_agent_config_with_custom_tools,
)
from searchagent.agents.tools.file_monitor import FileMonitor
from searchagent.agents.tools.pdf_search import PDFSearchTool

# TODO: Dynamically get directory path
input_dir = ""


AgentFunction = Callable[[], Union[str, "Agent", dict]]


# ========== Handleoff functions ==========


def transfer_back_to_triage():
    return triage_agent


def transfer_to_file_retrieval():
    return file_retrieval_agent


def transfer_to_sync():
    return sync_agent


def transfer_to_index():
    return index_agent


def transfer_to_embed():
    return embed_agent


# ========== Handleoff functions ==========


class Agent(BaseModel):
    name: str
    agent_config: QuickToolConfig
    functions: List[AgentFunction] = []


"""
TriageAgent serves as the central decision-maker
within the multi-agent orchestration system.
Its primary role is to analyze and route incoming tasks
or queries to the appropriate specialized agents or tools
based on the interpreted intent and extracted parameters.
"""
triage_agent = Agent(
    name="TriageAgent",
    agent_config=make_agent_config_with_custom_tools(
        tool_config=QuickToolConfig(tool_definitions=[], custom_tools=[]),
    ),
    functions=[transfer_to_file_retrieval],
)

"""Query Vector DB, i.e. pgvector"""
file_retrieval_agent = Agent(
    name="FileRetrievalAgent",
    agent_config=make_agent_config_with_custom_tools(
        tool_config=QuickToolConfig(
            tool_definitions=[], custom_tools=[PDFSearchTool(input_dir=input_dir)]
        ),
    ),
    functions=[transfer_back_to_triage],
)

"""Monitor file system changes and manage sync between local and db"""
sync_agent = Agent(
    name="SyncAgent",
    agent_config=make_agent_config_with_custom_tools(
        tool_config=QuickToolConfig(
            tool_definitions=[],
            custom_tools=[FileMonitor()],
            prompt_format="function_tag",
        ),
    ),
    functions=[transfer_back_to_triage, transfer_to_index, transfer_to_embed],
)


"""Handle indexing of the new or updated files using HNSW"""
index_agent = Agent(
    name="IndexAgent",
    agent_config=make_agent_config_with_custom_tools(
        tool_config=QuickToolConfig(
            tool_definitions=[], custom_tools=[], prompt_format="function_tag"
        ),
    ),
    functions=[transfer_back_to_triage],
)


"""Generate embeddings for new or updated files using ColPali"""
embed_agent = Agent(
    name="EmbedAgent",
    agent_config=make_agent_config_with_custom_tools(
        tool_config=QuickToolConfig(
            tool_definitions=[], custom_tools=[], prompt_format="function_tag"
        ),
    ),
    functions=[transfer_back_to_triage],
)
