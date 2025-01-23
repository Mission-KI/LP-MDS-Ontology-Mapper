from abc import ABC, abstractmethod
from logging import Logger
from pathlib import Path
from typing import Callable, Concatenate


class TaskContext(ABC):
    """A context provides a logger and supports executing sub-tasks."""

    def __init__(self, logger: Logger, working_path: Path):
        self._logger = logger
        self._working_path = working_path

    @property
    def logger(self) -> Logger:
        """Return logger for this context."""
        return self._logger

    @property
    def working_path(self) -> Path:
        """Return working path (not guaranteed that it exists, normally temp directory with write access)."""
        return self._working_path

    @property
    def input_path(self) -> Path:
        """Return path for input files (not guaranteed that it exists)."""
        return self._working_path / "input"

    @property
    def output_path(self) -> Path:
        """Return path for output files (not guaranteed that it exists)."""
        return self._working_path / "output"

    @abstractmethod
    def exec[**P, R](self, task_fn: Callable[Concatenate["TaskContext", P], R], *args: P.args, **kwargs: P.kwargs) -> R:
        """Execute a subtask. This creates a sub-context."""


class SimpleTaskContext(TaskContext):
    """Minimal task context."""

    def __init__(self, logger: Logger, working_path: Path):
        super().__init__(logger, working_path)

    def exec[**P, R](self, task_fn: Callable[Concatenate[TaskContext, P], R], *args: P.args, **kwargs: P.kwargs) -> R:
        new_logger = self.logger.getChild(task_fn.__name__)
        new_context = SimpleTaskContext(new_logger, self._working_path)
        return task_fn(new_context, *args, **kwargs)
