import logging

from langchain_core.tools import tool

from store.dealership_store import INVENTORY, Car

logger = logging.getLogger("ai-chat")


@tool
def list_inventory() -> list[Car]:
    """List available inventory

    Returns:
        Inventory: The inventory
    """
    logger.info("Getting inventory")
    return INVENTORY.availables
