from datetime import datetime
from enum import StrEnum, auto
from typing import List

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
    date: datetime
    car: Car
    name: str
    driver_licence: str


def create_initial_stock() -> Inventory:
    return Inventory(
        availables=[
            Car(model=Models.GOLF, color=Color.BLACK, kms=0, year=2025, value=35000),
            Car(model=Models.GOLF, color=Color.BLUE, kms=0, year=2025, value=35500),
            Car(model=Models.POLO, color=Color.WHITE, kms=0, year=2025, value=25000),
            Car(model=Models.POLO, color=Color.RED, kms=0, year=2025, value=25500),
            Car(model=Models.T_CROSS, color=Color.BLACK, kms=0, year=2025, value=33000),
            Car(
                model=Models.GOLF, color=Color.WHITE, kms=14867, year=2023, value=23500
            ),
            Car(
                model=Models.POLO, color=Color.BLACK, kms=58276, year=2020, value=12350
            ),
            Car(model=Models.POLO, color=Color.RED, kms=9239, year=2022, value=20000),
            Car(
                model=Models.T_CROSS,
                color=Color.BLACK,
                kms=67890,
                year=2024,
                value=27000,
            ),
        ]
    )


INVENTORY: Inventory = create_initial_stock()
TEST_DRIVE: List[TestDrive] = []
