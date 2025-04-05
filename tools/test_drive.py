import logging
from datetime import datetime
from random import randint
from typing import List

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from store.dealership_store import TEST_DRIVE, Car, TestDrive

logger = logging.getLogger("ai-chat")


class CancelTestDrive(BaseModel):
    """Cancel a test drive"""

    code: int = Field(description="The code of the test drive to cancel")


def cancel_test_drive(code: int) -> bool:
    """Cance a test drive

    Args:
        code (int): The code of the test drive

    Returns:
        bool: Confirm the cancel
    """
    logger.info(f"Cancel test drive of code {code}")
    for i, test in enumerate(TEST_DRIVE):
        if test.code == code:
            TEST_DRIVE.pop(i)
            return True
    return False


@tool
def schedule_test_drive(
    date: datetime, car: Car, name: str, driver_licence: str
) -> int:
    """Schedule a test drive for a specific car

    Args:
        date (datetime): Date and time for the test driver
        car (str): The car that the client wants to test
        name (str): Client's name
        driver_licence (str): Client's driver licence

    Returns:
        int: Test drive code
    """
    code = randint(0, 10)
    test_drive = TestDrive(
        code=code,
        date=date,
        car=car,
        name=name,
        driver_licence=driver_licence,
    )
    logger.info(f"Scheduling test drive: {test_drive}")
    TEST_DRIVE.append(test_drive)
    return code


@tool
def list_test_drives() -> List[TestDrive]:
    """List the scheduled test drivers

    Returns:
        List[TestDrive]: A list of the scheduled test drivers
    """
    logger.info("Returning all test drivers")
    return TEST_DRIVE
