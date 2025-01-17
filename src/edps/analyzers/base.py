from abc import ABC, abstractmethod
from typing import AsyncIterator

from edps.task import TaskContext
from edps.types import DataSet


class Analyzer(ABC):
    @abstractmethod
    def analyze(self, ctx: TaskContext) -> AsyncIterator[DataSet]: ...
