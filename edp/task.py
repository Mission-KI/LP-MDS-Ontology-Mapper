from abc import ABC, abstractmethod
from logging import Logger
from typing import Callable, Concatenate

from edp.context import OutputContext


# TODO: Does it make sense to combine this with OutputContext and JobRepository/JobSession and even Config?
# TODO: Maybe TaskContext should contain OutputContext and JobSession (because persistent version needs to persist logs)?
class TaskContext(ABC):
    """A context provides a logger and supports executing sub-tasks."""

    def __init__(self, logger: Logger, output_context: OutputContext):
        self._logger = logger
        self._output_context = output_context

    @property
    def logger(self) -> Logger:
        """Return logger for this context."""
        return self._logger

    @property
    def output_context(self) -> OutputContext:
        """Return output context for this context."""
        return self._output_context

    @abstractmethod
    def exec[**P, R](self, task_fn: Callable[Concatenate["TaskContext", P], R], *args: P.args, **kwargs: P.kwargs) -> R:
        """Execute a subtask. This creates a sub-context."""


class SimpleTaskContext(TaskContext):
    """Minimal task context."""

    def __init__(self, logger: Logger, output_context: OutputContext):
        super().__init__(logger, output_context)

    def exec[**P, R](self, task_fn: Callable[Concatenate[TaskContext, P], R], *args: P.args, **kwargs: P.kwargs) -> R:
        new_logger = self.logger.getChild(task_fn.__name__)
        new_context = SimpleTaskContext(new_logger, self.output_context)
        return task_fn(new_context, *args, **kwargs)
