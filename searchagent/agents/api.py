from searchagent.agents.common.client_utils import (
    QuickToolConfig,
    make_agent_config_with_custom_tools,
)
from searchagent.agents.tools.file_monitor import FileMonitor
from searchagent.agents.tools.pdf_search import PDFSearchTool


def transfer_back_to_triage():
    return TriageAgent


def transfer_to_file_retrieval():
    return FileRetrievalAgent


def transfer_to_index():
    return IndexAgent


def transfer_to_embed():
    return EmbedAgent


def transfer_to_sync():
    return SyncAgent


class TriageAgent:
    """
    TriageAgent serves as the central decision-maker
    within the multi-agent orchestration system.
    Its primary role is to analyze and route incoming tasks
    or queries to the appropriate specialized agents or tools
    based on the interpreted intent and extracted parameters.
    """

    agent_config = make_agent_config_with_custom_tools(
        disable_safety=True,
        tool_config=QuickToolConfig(
            tool_definitions=[],
            custom_tools=[
                transfer_to_file_retrieval,
                transfer_to_index,
                transfer_to_embed,
                transfer_to_sync,
            ],
        ),
    )


class FileRetrievalAgent:
    agent_config = make_agent_config_with_custom_tools(
        disable_safety=True,
        tool_config=QuickToolConfig(
            tool_definitions=[],
            custom_tools=[PDFSearchTool],
        ),
    )


class SyncAgent:
    agent_config = make_agent_config_with_custom_tools(
        disable_safety=True,
        tool_config=QuickToolConfig(
            tool_definitions=[],
            custom_tools=[FileMonitor],
        ),
    )


class IndexAgent:
    pass


class EmbedAgent:
    pass
