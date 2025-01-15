from abc import ABC, abstractmethod
from typing import Union

from edps.task import TaskContext
from edps.types import DataSetType, StructuredDataSet


class Analyzer(ABC):
    @property
    @abstractmethod
    def data_set_type(self) -> DataSetType: ...
    @abstractmethod
    async def analyze(self, ctx: TaskContext) -> Union[StructuredDataSet]: ...
