import logging
from datetime import datetime
from typing import List

from langchain_core.tools import tool

from store.dealership_store import TEST_DRIVE, Car, TestDrive

logger = logging.getLogger(__name__)


@tool
def schedule_test_drive(
    date: datetime, car: Car, name: str, driver_licence: str
) -> bool:
    """Schedule a test drive for a specific car

    Args:
        date (datetime): Date and time for the test driver
        car (str): The car that the client wants to test
        name (str): Client's name
        driver_licence (str): Client's driver licence

    Returns:
        bool: Confirm the schedule
    """
    test_drive = TestDrive(date=date, car=car, name=name, driver_licence=driver_licence)
    print(f"Scheduling an test drive: {test_drive}")
    TEST_DRIVE.append(test_drive)
    return True


@tool
def list_test_drives() -> List[TestDrive]:
    """List the scheduled test drivers

    Returns:
        List[TestDrive]: A list of the scheduled test drivers
    """
    print("Returning test drivers")
    return TEST_DRIVE
