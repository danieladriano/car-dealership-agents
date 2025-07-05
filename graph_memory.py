from pathlib import Path

from cognee import SearchType, config, search


class GraphMemory:
    def __init__(self, data_path: Path, cognee_path: Path) -> None:
        self._load_cognee(data_path=data_path, cognee_path=cognee_path)

    def _load_cognee(self, data_path: Path, cognee_path: Path) -> None:
        config.data_root_directory(data_root_directory=str(data_path))
        config.system_root_directory(system_root_directory=str(cognee_path))

    async def search(self, query_text: str) -> str:
        response = await search(
            query_text=query_text, query_type=SearchType.RAG_COMPLETION
        )
        return response[0]
