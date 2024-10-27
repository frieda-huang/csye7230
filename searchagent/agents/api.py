from searchagent.agents.common.factory import AgentFactory
from searchagent.agents.tools.routines.embed import Embed
from searchagent.agents.tools.routines.file_monitor import FileMonitor
from searchagent.agents.tools.routines.index import Index
from searchagent.agents.tools.routines.pdf_search import PDFSearchTool
from searchagent.agents.common.types import AgentType

# TODO: Dynamically get directory path
input_dir = "."


class AgentManager:
    def __init__(self, factory: AgentFactory):
        self.factory = factory

    async def create_agents(self):
        """TriageAgent routes incoming queries to the appropriate specialized agents or tools"""
        await self.factory.create_and_register_agent(
            name=AgentType.triage_agent, custom_tools=[]
        )

        """Query pgvector database"""
        await self.factory.create_and_register_agent(
            name=AgentType.file_retrieval_agent,
            custom_tools=[PDFSearchTool(input_dir=input_dir)],
        )

        """Monitor file system changes and manage sync between local and db"""
        await self.factory.create_and_register_agent(
            name=AgentType.sync_agent, custom_tools=[FileMonitor()]
        )

        """Handle indexing of the new or updated files using HNSW"""
        await self.factory.create_and_register_agent(
            name=AgentType.index_agent, custom_tools=[Index()]
        )

        """Generate embeddings for new or updated files using ColPali"""
        await self.factory.create_and_register_agent(
            name=AgentType.embed_agent, custom_tools=[Embed()]
        )
