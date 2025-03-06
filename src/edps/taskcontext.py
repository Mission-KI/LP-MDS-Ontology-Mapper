from abc import ABC, abstractmethod
from logging import Logger
from pathlib import Path, PurePosixPath
from typing import Awaitable, Callable, Concatenate, Iterator, Optional, Unpack, overload

from extended_dataset_profile.models.v0.edp import DataSet


class TaskContext(ABC):
    """Interface. A context provides a logger and supports executing sub-tasks."""

    @property
    @abstractmethod
    def logger(self) -> Logger:
        """Return logger for this context."""

    @property
    @abstractmethod
    def input_path(self) -> Path:
        """Return path for input files (common to all TaskContexts in hierarchy)."""

    @property
    @abstractmethod
    def output_path(self) -> Path:
        """Return path for output files (common to all TaskContexts in hierarchy)."""

    @abstractmethod
    def create_working_dir(self, name: str) -> Path:
        """Create a working directory specific to this TaskContext."""

    @property
    @abstractmethod
    def children(self) -> list["TaskContext"]:
        """Return all child TaskContexts."""

    @property
    @abstractmethod
    def name_parts(self) -> list[str]:
        """Return all name parts coming from the TaskContext hierarchy."""

    @property
    @abstractmethod
    def qualified_path(self) -> PurePosixPath:
        """Return a qualified path containing of all the name parts."""

    @property
    @abstractmethod
    def dataset_name(self) -> Optional[str]:
        """Return the last name part identifying the dataset."""

    @property
    @abstractmethod
    def dataset(self) -> Optional[DataSet]:
        """Return the DataSet attached to the TaskContext."""

    @abstractmethod
    def build_output_reference(self, final_part: str) -> str:
        """Build a new output reference that can be used for a file name in the output path consisting of all the name parts."""

    @abstractmethod
    def relative_path(self, path: Path) -> Path:
        """Convert the input path to a path relative to the TaskContext base path."""

    @abstractmethod
    def collect_datasets(self) -> Iterator[DataSet]:
        """Return the datasets of this TaskContext and all its children using depth-first traversal."""

    @abstractmethod
    async def exec[**P, R_DS: DataSet, *R_Ts](
        self,
        dataset_name: str,
        task_fn: Callable[Concatenate["TaskContext", P], Awaitable[R_DS] | Awaitable[tuple[R_DS, Unpack[R_Ts]]]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        """Execute a subtask swallowing the results of the task function. Any occuring errors are caught and logged."""

    @overload
    async def exec_with_result[**P, R_DS: DataSet, *R_Ts](
        self,
        dataset_name: str,
        task_fn: Callable[Concatenate["TaskContext", P], Awaitable[tuple[R_DS, Unpack[R_Ts]]]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> tuple[R_DS, Unpack[R_Ts]]:
        """Execute a subtask returning a tuple of DataSet and other information. This creates a sub-context."""

    @overload
    async def exec_with_result[**P, R_DS: DataSet](
        self,
        dataset_name: str,
        task_fn: Callable[Concatenate["TaskContext", P], Awaitable[R_DS]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R_DS:
        """Execute a subtask returning just a DataSet. This creates a sub-context."""

    @abstractmethod
    async def exec_with_result[**P, R](
        self,
        dataset_name: str,
        task_fn: Callable[Concatenate["TaskContext", P], Awaitable[R]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Execute a subtask returning the results of the task function. This creates a sub-context."""

    @abstractmethod
    async def import_file(self, dataset_name: str, path: Path) -> None:
        """Import and analyze the file if it's a supported type."""
