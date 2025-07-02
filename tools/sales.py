import logging
from typing import Optional

from click import Option
from langchain_core.tools import tool

from store.dealership_store import INVENTORY, Car, Models

logger = logging.getLogger("ai-chat")


@tool
def list_inventory() -> list[Car]:
    """List available inventory

    Returns:
        Inventory: The inventory
    """
    logger.info("Getting inventory")
    return INVENTORY.availables


@tool
def car_information(model: Models, year: int) -> Optional[Car]:
    """Get more information about specific car

    Args:
        model (Models): Car model to get information
        year (int): Car year to get information

    Returns:
        Optional[Car]: The car that the user wants more information
    """
    for car in INVENTORY.availables:
        if car.model == model and car.year == year:
            logger.info(f"Car information {car}")
            return car
    logger.info("The car was not found")
    return None
