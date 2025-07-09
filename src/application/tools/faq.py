from pydantic import BaseModel, Field


class CarModelDetails(BaseModel):
    """
    If the user asks for more information about a specific car model.
    You can awser questions about:
    - The engine
    - safety features
    - dimensions
    """

    user_request: str = Field(
        description="The user request about the car model. Do not specify the year of the car."
    )
