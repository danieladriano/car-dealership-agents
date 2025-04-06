from datetime import datetime
from enum import StrEnum, auto
from typing import List

import orjson
from pydantic import BaseModel


class Brand(StrEnum):
    VW = "Volkswagen"


class Models(StrEnum):
    GOLF = auto()
    POLO = auto()
    T_CROSS = auto()


class Color(StrEnum):
    BLACK = auto()
    BLUE = auto()
    WHITE = auto()
    RED = auto()
    GREEN = auto()


class TestDriveStatus(StrEnum):
    SCHEDULED = auto()
    CANCEL = auto()
    DONE = auto()


class Car(BaseModel):
    brand: Brand = Brand.VW
    model: Models
    color: Color
    kms: int
    year: int
    value: float


class Inventory(BaseModel):
    availables: list[Car]
    out_of_stock: list[Car] = []


class TestDrive(BaseModel):
    code: int
    date: datetime
    car: Car
    name: str
    driver_licence: str
    status: TestDriveStatus = TestDriveStatus.SCHEDULED


def save_inventory() -> None:
    with open("./store/inventory.json", "w") as f:
        f.write(INVENTORY.model_dump_json(indent=4))


def save_test_drivers() -> None:
    with open("./store/test_driver.json", "wb") as f:
        f.write(orjson.dumps([test_drive.model_dump() for test_drive in TEST_DRIVE]))


def load_inventory() -> Inventory:
    with open("./store/inventory.json", "rb") as f:
        row_data = orjson.loads(f.read())
    return Inventory(**row_data)


def load_test_drivers() -> list[TestDrive]:
    with open("./store/test_driver.json", "rb") as f:
        row_data = orjson.loads(f.read())
    return [TestDrive(**data) for data in row_data]


INVENTORY: Inventory = load_inventory()
TEST_DRIVE: List[TestDrive] = load_test_drivers()
