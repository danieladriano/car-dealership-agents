from langchain_core.tools import tool

from store.dealership_store import INVENTORY, Car


@tool
def list_inventory() -> list[Car]:
    """List available inventory

    Returns:
        Inventory: The inventory
    """
    print("Getting inventory")
    return INVENTORY.availables
