from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from edps.types import Config

# Model for inputs/algoCustomData.json


class CustomDataFileInfo(BaseModel):
    fileExtension: str


class CustomData(BaseModel):
    """Data that is not Pontus-X standard, but needed by the service."""

    fileInfo: CustomDataFileInfo = Field(description="Information about the file getting analyzed.")
    config: Config = Field(
        default_factory=lambda: Config(), description="Configuration of the analysis steps. These are optional."
    )

    @staticmethod
    def read_from_json_file(json_path: Path) -> "CustomData":
        return CustomData.model_validate_json(json_path.read_text())


# Model for ddos/DID (in JSON format)
# See DDO spec: https://docs.oceanprotocol.com/developers/ddo-specification


class DDOMetadata(BaseModel):
    created: Optional[datetime]
    updated: Optional[datetime]
    name: str
    description: str
    author: str
    license: str
    tags: list[str] = []


class DDO(BaseModel):
    id: str
    metadata: DDOMetadata


def read_ddo_file(json_file: Path):
    with open(json_file, "r") as file:
        json_data = file.read()
    return DDO.model_validate_json(json_data)
