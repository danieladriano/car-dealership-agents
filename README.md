# Car dealership

In this Agent exemple, we use a short-term memory to help our agent to remmenber previous interactions with the user in the same thread-id. Also, we use a interrupt to ask the user for confirmatiom. The agent are using a local LLM running using ollama.
To run the project:

## Getting started

To locally run the [qwen2.5:14b](https://ollama.com/library/qwen2.5:14b), we are using [Ollama](https://ollama.com/download/linux) version 0.5.7

Pull qwen2.5:14b
```
ollama pull qwen2.5:14b
```

The project uses [uv](https://docs.astral.sh/uv/) version 0.6.5 as a dependency management tool.

Create a development environment:
```
uv sync
```

Run the project:
```
uv run main.py
```