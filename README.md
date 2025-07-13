# Car Dealership Agent

A car dealership agent who can use Gemini or models hosted on Ollama to run. Use Cognee to create a graph memory for our agent, which holds some FAQ files.

The agent can list the vehicles available in stock, schedule, list, or cancel a test drive, and also search in the graph memory (FAQ) for more technical information about some car models (Golf, Polo, and T-Cross).

## Getting Started

### Prerequisites

It is necessary to create a `.env` file that holds some configurations and your Google API key. Use `.env-template`.

### Dependencies

The project uses [uv](https://docs.astral.sh/uv/) version 0.7.9 as a dependency management tool.

Create a development environment:
```
uv sync
```

## Makefile Commands

The following commands are available in the `Makefile`:

* `make create-graph files="<file_list>"`: Creates a graph from the specified files. For example:
```
make create-graph files="data/faq/"
```
* `make search-graph`: Runs a streamlit app that searches inside the memory graph created.
* `make run-agent`: Runs the car dealership agent.
* `make format-lit-fix`: format the code