from abc import ABC, abstractmethod
from typing import Union

from edp.context import OutputContext
from edp.task import TaskContext
from edp.types import DataSetType, StructuredDataSet


class Analyzer(ABC):
    @property
    @abstractmethod
    def data_set_type(self) -> DataSetType: ...
    @abstractmethod
    async def analyze(self, ctx: TaskContext, output_context: OutputContext) -> Union[StructuredDataSet]: ...
