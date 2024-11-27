import json

from anthropic import Anthropic
from claude_agent.tools import ToolNames, colpali_embed, colpali_search, tools
from searchagent.utils import project_paths

client = Anthropic()
MODEL_NAME = "claude-3-opus-20240229"


def process_tool_call(tool_name, tool_input):
    if tool_name == ToolNames.colpali_search:
        return colpali_search(
            tool_input["query"], tool_input["top_k"], tool_input["email"]
        )

    if tool_name == ToolNames.colpali_embed:
        return colpali_embed(tool_input["filepaths"])


def chat_with_claude(user_message):
    print(f"\n{'=' * 50}\nUser Message: {user_message}\n{'=' * 50}")

    message = client.messages.create(
        system="You will use the email colpalisearch@gmail.com when using the colpali_search tool."
        "You will patiently wait for the result to return.",
        model=MODEL_NAME,
        max_tokens=4096,
        messages=[{"role": "user", "content": user_message}],
        tools=tools,
    )

    print("\nInitial Response:")
    print(f"Stop Reason: {message.stop_reason}")
    print(f"Content: {message.content}")

    if message.stop_reason == "tool_use":
        tool_use = next(block for block in message.content if block.type == "tool_use")
        tool_name = tool_use.name
        tool_input = tool_use.input

        print(f"\nTool Used: {tool_name}")
        print(f"Tool Input: {tool_input}")

        tool_result = process_tool_call(tool_name, tool_input)

        print(f"Tool Result: {tool_result}")

        response = client.messages.create(
            model=MODEL_NAME,
            max_tokens=4096,
            messages=[
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": message.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": tool_result,
                        }
                    ],
                },
            ],
            tools=tools,
        )
    else:
        response = message

    final_response = next(
        (block.text for block in response.content if hasattr(block, "text")),
        None,
    )
    print(response.content)
    print(f"\nFinal Response: {final_response}")

    return final_response


# Example usage
filepaths = [
    f"{project_paths.SINGLE_FILE_DIR}/ColPali_Efficient_Document_Retrieval_with_Vision_Language_Models.pdf"
    f"{project_paths.PDF_DIR}/Attention_Is_All_You_Need.pdf"
]

chat_with_claude("Find a PDF page on Scaled Dot-Product Attention")
chat_with_claude("Find a PDF page on Multi-Head Attention")
chat_with_claude(f"Help me embed these PDF files: {json.dumps(filepaths)}")
