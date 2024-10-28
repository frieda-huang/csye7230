from enum import Enum

from llama_stack_client.types.tool_param_definition_param import (
    ToolParamDefinitionParam,
)


class AgentType(Enum):
    triage_agent = "triage_agent"
    file_retrieval_agent = "file_retrieval_agent"
    sync_agent = "sync_agent"
    index_agent = "index_agent"
    embed_agent = "embed_agent"

    def __str__(self):
        return self.value


# FIXME: This is a hack to bypass the JSON serialization error in custom tool's get_params_definition()
custom_tool_handoff_agent_params = {
    "query": ToolParamDefinitionParam(
        param_type="str",
        description="Determines which specialized agent or custom tool should handle the task",
        required=True,
    )
}
