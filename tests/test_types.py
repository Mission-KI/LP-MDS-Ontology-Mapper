from datetime import datetime, timedelta
from enum import Enum
from logging import getLogger
from pathlib import PurePosixPath
from types import NoneType
from typing import Union, get_args, get_origin

from pydantic import AnyUrl, BaseModel

from edp.types import ExtendedDatasetProfile

logger = getLogger(__name__)


async def test_validate_EDP_model():
    models_already_processed: set[type[BaseModel]] = set()
    models_to_process: list[type[BaseModel]] = [ExtendedDatasetProfile]

    while len(models_to_process) != 0:
        model = models_to_process.pop(0)
        if model not in models_already_processed:
            models_to_process.extend(_validate_model(model))
            models_already_processed.add(model)


def _validate_model(model_type: type[BaseModel]):
    logger.info(f"Checking model {model_type.__name__}")
    for field_name, field in model_type.__pydantic_fields__.items():
        field_type = field.annotation
        if field_type:
            try:
                yield from _validate_model_type(field_type)
            except TypeError as e:
                raise TypeError(f"{model_type.__name__}.{field_name}: Type {e} is not supported in the schema!")


# Check if the given type is valid for the EDPS model and return additional types that need checking.
def _validate_model_type(current_type: type):
    if current_type in [NoneType, str, int, float, complex, bool, datetime, timedelta, PurePosixPath, AnyUrl]:
        pass
    elif isinstance(current_type, type) and issubclass(current_type, Enum):
        pass
    elif isinstance(current_type, type) and issubclass(current_type, BaseModel):
        yield current_type
    elif get_origin(current_type) in [Union, list, set]:
        for generic_type in get_args(current_type):
            yield from _validate_model_type(generic_type)
    else:
        raise TypeError(str(current_type))
