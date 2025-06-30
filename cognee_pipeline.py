from argparse import ArgumentParser
import os
import asyncio
from pathlib import Path
from cognee import config, prune, add, cognify, visualize_graph


def config_cognee() -> None:
    data_directory_path = str(
        Path(os.path.join(Path(__file__).parent, ".data_storage")).resolve()
    )

    cognee_directory_path = str(
        Path(os.path.join(Path(__file__).parent, ".cognee_system")).resolve()
    )

    config.data_root_directory(data_directory_path)
    config.system_root_directory(cognee_directory_path)


async def prune_cognee() -> None:
    await prune.prune_data()
    await prune.prune_system(metadata=True)


async def main(assets_path: Path) -> None:
    config_cognee()
    await prune_cognee()

    for file_path in assets_path.iterdir():
        with open(file_path, "r") as file:
            file_content = file.read()
            if file_content:
                await add(file_content)

    await cognify()

    graph_file_path = str(
        Path(
            os.path.join(Path(__file__).parent, ".artifacts/graph_visualization.html")
        ).resolve()
    )
    await visualize_graph(graph_file_path)


if __name__ == "__main__":
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--files", "-f")
    args = argument_parser.parse_args()

    asyncio.run(main(assets_path=Path(args.files)))
