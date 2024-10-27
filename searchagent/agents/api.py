from enum import Enum

from searchagent.agents.common.factory import AgentFactory
from searchagent.agents.tools.routines.embed import Embed
from searchagent.agents.tools.routines.file_monitor import FileMonitor
from searchagent.agents.tools.routines.index import Index
from searchagent.agents.tools.routines.pdf_search import PDFSearchTool

# TODO: Dynamically get directory path
input_dir = "."


class Agent(Enum):
    TRIAGE = "TriageAgent"
    RETRIEVAL = "FileRetrievalAgent"
    SYNC = "SyncAgent"
    INDEX = "IndexAgent"
    EMBED = "EmbedAgent"

    def __str__(self):
        return self.value


class AgentManager:
    def __init__(self, factory: AgentFactory):
        self.factory = factory

    def transfer_back_to_triage(self):
        return self.factory.get_agent(Agent.TRIAGE)

    def transfer_to_file_retrieval(self):
        return self.factory.get_agent(Agent.RETRIEVAL)

    def transfer_to_sync(self):
        return self.factory.get_agent(Agent.SYNC)

    def transfer_to_index(self):
        return self.factory.get_agent(Agent.INDEX)

    def transfer_to_embed(self):
        return self.factory.get_agent(Agent.EMBED)

    async def create_agents(self):
        """TriageAgent routes incoming queries to the appropriate specialized agents or tools"""
        await self.factory.create_and_register_agent(name=Agent.TRIAGE)

        """Query Vector DB, i.e. pgvector"""
        await self.factory.create_and_register_agent(
            name=Agent.RETRIEVAL, custom_tools=[PDFSearchTool(input_dir=input_dir)]
        )

        """Monitor file system changes and manage sync between local and db"""
        await self.factory.create_and_register_agent(
            name=Agent.SYNC, custom_tools=[FileMonitor()]
        )

        """Handle indexing of the new or updated files using HNSW"""
        await self.factory.create_and_register_agent(
            name=Agent.INDEX, custom_tools=[Index()]
        )

        """Generate embeddings for new or updated files using ColPali"""
        await self.factory.create_and_register_agent(
            name=Agent.EMBED, custom_tools=[Embed()]
        )
