from abc import ABC, abstractmethod
from typing import AsyncIterator

from extended_dataset_profile.models.v0.edp import DataSet

from edps.task import TaskContext


class Analyzer(ABC):
    @abstractmethod
    def analyze(self, ctx: TaskContext) -> AsyncIterator[DataSet]: ...
