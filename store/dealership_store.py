from pydantic import BaseModel
from enum import StrEnum, auto

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


class Inventory(BaseModel):
    availables: list[Car]
    out_of_stock: list[Car] = []


def create_initial_stock() -> Inventory:
    return Inventory(
        availables=[
            Car(model=Models.GOLF, color=Color.BLACK, kms=0, year=2025),
            Car(model=Models.GOLF, color=Color.BLUE, kms=0, year=2025),
            Car(model=Models.POLO, color=Color.WHITE, kms=0, year=2025),
            Car(model=Models.POLO, color=Color.RED, kms=0, year=2025),
            Car(model=Models.T_CROSS, color=Color.BLACK, kms=0, year=2025),
            Car(model=Models.GOLF, color=Color.WHITE, kms=14867, year=2023),
            Car(model=Models.POLO, color=Color.BLACK, kms=58276, year=2020),
            Car(model=Models.POLO, color=Color.RED, kms=9239, year=2022),
            Car(model=Models.T_CROSS, color=Color.BLACK, kms=67890, year=2024),
        ]
    )


INVENTORY: Inventory = create_initial_stock()
