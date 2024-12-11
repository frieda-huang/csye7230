import typer
from claude_agent.base import chat_with_claude

app = typer.Typer()


@app.command()
def file_assistant(query: str):
    chat_with_claude(query)


if __name__ == "__main__":
    app()
