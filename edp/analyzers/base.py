from abc import ABC, abstractmethod

from edp.context import OutputContext
from edp.types import Dataset, DataSetType


class Analyzer(ABC):
    @property
    @abstractmethod
    def data_set_type(self) -> DataSetType: ...
    @abstractmethod
    async def analyze(self, output_context: OutputContext) -> Dataset: ...
