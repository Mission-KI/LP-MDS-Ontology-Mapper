import asyncio
import re
import warnings
from logging import Logger
from pathlib import Path, PurePosixPath
from typing import Awaitable, Callable, Concatenate, Iterator, Optional, Unpack, cast
from uuid import UUID

from extended_dataset_profile.models.v0.edp import _BaseDataSet

from edps.file import determine_file_type, sanitize_path
from edps.importers import lookup_importer, lookup_unsupported_type_message
from edps.taskcontext import TaskContext
from edps.types import DataSet


class TaskContextImpl(TaskContext):
    """A context provides a logger and supports executing sub-tasks."""

    def __init__(self, logger: Logger, base_path: Path, name_parts: list[str] = []):
        self._logger = logger
        self._name_parts = name_parts
        self._children: list[TaskContext] = []
        self._dataset: Optional[DataSet] = None

        self._base_path = base_path.resolve()
        if not base_path.is_dir():
            raise FileNotFoundError(f"Path '{base_path}' doesn't exist or is not a directory!")
        self._input_path = base_path / "input"
        self._input_path.mkdir(exist_ok=True)
        self._output_path = base_path / "output"
        self._output_path.mkdir(exist_ok=True)

    @property
    def logger(self) -> Logger:
        """Return logger for this context."""
        return self._logger

    @property
    def input_path(self) -> Path:
        """Return path for input files (common to all TaskContexts in hierarchy)."""
        return self._input_path

    @property
    def output_path(self) -> Path:
        """Return path for output files (common to all TaskContexts in hierarchy)."""
        return self._output_path

    def create_working_dir(self, name: str) -> Path:
        """Create a working directory specified by this TaskContext and the provided name."""
        working_path = self._base_path / "work" / "_".join(self._name_parts) / name
        working_path.mkdir(parents=True)
        return working_path

    @property
    def children(self) -> list["TaskContext"]:
        return self._children

    @property
    def name_parts(self) -> list[str]:
        return self._name_parts

    @property
    def qualified_path(self) -> PurePosixPath:
        return PurePosixPath("/".join(self._name_parts))

    def build_output_reference(self, final_part: str) -> PurePosixPath:
        ref = "_".join(self._name_parts) + "_" + final_part
        return PurePosixPath(re.sub("[./]", "_", ref))

    def relative_path(self, path: Path) -> Path:
        return path.relative_to(self._base_path)

    @property
    def dataset_name(self) -> Optional[str]:
        return self.name_parts[-1] if self.name_parts else None

    @property
    def dataset(self) -> Optional[DataSet]:
        return self._dataset

    def collect_datasets(self) -> Iterator[DataSet]:
        parent_uuid: Optional[UUID] = None
        if own_dataset := self.dataset:
            parent_uuid = own_dataset.uuid
            yield own_dataset

        for child in self.children:
            if direct_child_ds := child.dataset:
                direct_child_ds.parentUuid = parent_uuid
            yield from child.collect_datasets()

    async def exec[**P, R_DS: DataSet, *R_Ts](
        self,
        dataset_name: str,
        task_fn: Callable[Concatenate["TaskContext", P], Awaitable[R_DS] | Awaitable[tuple[R_DS, Unpack[R_Ts]]]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> None:
        """Execute a subtask, store the dataset and swallow the results. Catch and log any errors."""

        try:
            await self.exec_with_result(dataset_name, task_fn, *args, **kwargs)
        except Exception as exception:
            self.logger.error("Error in sub context, continuing anyways...", exc_info=exception)
            warnings.warn(f"Error in sub context: {exception}")
            return

    async def exec_with_result[**P, R](
        self,
        dataset_name: str,
        task_fn: Callable[Concatenate["TaskContext", P], Awaitable[R]],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> R:
        """Execute a subtask, store the dataset and return the results."""

        # Replace dot with underscore. Keep slash as they are needed in archives.
        dataset_name = sanitize_path(dataset_name.replace(".", "_"))
        child_context = self._prepare_sub_context(dataset_name)

        try:
            result = await task_fn(child_context, *args, **kwargs)
        except Exception as exception:
            child_context.logger.error("Error", exc_info=exception)
            raise exception

        if isinstance(result, _BaseDataSet):
            child_context._put_dataset(cast(DataSet, result), dataset_name)
        elif isinstance(result, tuple) and isinstance(result[0], _BaseDataSet):
            child_context._put_dataset(cast(DataSet, result[0]), dataset_name)
        else:
            child_context.logger.error("Task function didn't return a dataset.")
            raise RuntimeError("Task function didn't return a dataset.")
        return result

    async def import_file(self, path: Path, dataset_name: Optional[str] = None) -> None:
        """Import file if supported. Store the dataset. Catch and log any errors."""
        if dataset_name is None:
            dataset_name = path.name

        if path.is_dir():
            async with asyncio.TaskGroup() as group:
                for sub_file in path.iterdir():
                    group.create_task(dir_context.import_file(sub_file.name, sub_file))
            return

        file_type = determine_file_type(path)
        importer = lookup_importer(file_type)
        if importer is None:
            message = lookup_unsupported_type_message(file_type)
            self.logger.warning(message)
            warnings.warn(message, RuntimeWarning)
        else:
            await self.exec(dataset_name, importer, path)

    async def import_file_with_result(self, path: Path, dataset_name: Optional[str] = None) -> DataSet:
        """Import file if supported. Store and return the dataset."""
        if path.is_dir():
            raise RuntimeError("Can not run the import_file_with_results() on directories. Use import_file instead!")

        file_type = determine_file_type(path)
        importer = lookup_importer(file_type)
        if importer is None:
            message = lookup_unsupported_type_message(file_type)
            raise NotImplementedError(message)
        else:
            return await self.exec_with_result(dataset_name, importer, path)

    def _prepare_sub_context(self, dataset_name: str) -> "TaskContextImpl":
        child_name_parts = self.name_parts + [dataset_name]
        new_logger = self.logger.getChild(dataset_name)
        sub_context = TaskContextImpl(
            new_logger,
            self._base_path,
            child_name_parts,
        )
        self.children.append(sub_context)
        return sub_context

    def _put_dataset(self, dataset: DataSet, dataset_name: str):
        if self._dataset:
            raise RuntimeError(
                "There is already a dataset in this context. You need to put exactly one dataset into this dataset context!"
            )
        dataset.name = dataset_name
        self._dataset = dataset
