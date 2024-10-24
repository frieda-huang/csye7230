from functools import partial

from searchagent.agents.factory import AgentFactory
from searchagent.agents.tools.embed import Embed
from searchagent.agents.tools.file_monitor import FileMonitor
from searchagent.agents.tools.index import Index
from searchagent.agents.tools.pdf_search import PDFSearchTool

# TODO: Dynamically get directory path
input_dir = "."


def transfer_back_to_triage(factory: AgentFactory):
    return factory.get_agent("TriageAgent")


def transfer_to_file_retrieval(factory: AgentFactory):
    return factory.get_agent("FileRetrievalAgent")


def transfer_to_sync(factory: AgentFactory):
    return factory.get_agent("SyncAgent")


def transfer_to_index(factory: AgentFactory):
    return factory.get_agent("IndexAgent")


def transfer_to_embed(factory: AgentFactory):
    return factory.get_agent("EmbedAgent")


async def create_agents(factory: AgentFactory):
    """TriageAgent routes incoming queries to the appropriate specialized agents or tools"""
    await factory.create_and_register_agent(
        name="TriageAgent", functions=[partial(transfer_to_file_retrieval, factory)]
    )

    """Query Vector DB, i.e. pgvector"""
    await factory.create_and_register_agent(
        name="FileRetrievalAgent",
        functions=[partial(transfer_back_to_triage, factory)],
        custom_tools=[PDFSearchTool(input_dir=input_dir)],
    )

    """Monitor file system changes and manage sync between local and db"""
    await factory.create_and_register_agent(
        name="SyncAgent",
        functions=[
            partial(transfer_back_to_triage, factory),
            partial(transfer_to_index, factory),
            partial(transfer_to_embed, factory),
        ],
        custom_tools=[FileMonitor()],
    )

    """Handle indexing of the new or updated files using HNSW"""
    await factory.create_and_register_agent(
        name="IndexAgent",
        functions=[partial(transfer_back_to_triage, factory)],
        custom_tools=[Index()],
    )

    """Generate embeddings for new or updated files using ColPali"""
    await factory.create_and_register_agent(
        name="EmbedAgent",
        functions=[partial(transfer_back_to_triage, factory)],
        custom_tools=[Embed()],
    )
