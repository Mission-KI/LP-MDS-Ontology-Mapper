from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import BaseModel

# Model for inputs/algoCustomData.json


class CustomDataFileInfo(BaseModel):
    fileExtension: str


class CustomData(BaseModel):
    fileInfo: CustomDataFileInfo


def read_custom_data_file(json_file: Path):
    with open(json_file, "r") as file:
        json_data = file.read()
    return CustomData.model_validate_json(json_data)


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
