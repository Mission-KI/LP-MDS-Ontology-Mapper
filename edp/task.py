from abc import ABC, abstractmethod
from logging import Logger
from typing import Callable, Concatenate


# TODO: Does it make sense to combine this with OutputContext and JobRepository/JobSession?
# TODO: Maybe TaskContext should contain OutputContext and JobSession (because persistent version needs to persist logs)?
class TaskContext(ABC):
    """A context provides a logger and supports executing sub-tasks."""

    def __init__(self, logger: Logger):
        self._logger = logger

    @property
    def logger(self) -> Logger:
        """Return logger for this context."""
        return self._logger

    @abstractmethod
    def exec[**P, R](
        self, task_fn: Callable[Concatenate["TaskContext", P], R], *args: P.args, **kwargs: P.kwargs
    ) -> R:
        """Execute a subtask. This creates a sub-context."""


class SimpleTaskContext(TaskContext):
    """Minimal task context."""

    def __init__(self, logger: Logger):
        super().__init__(logger)

    def exec[**P, R](self, task_fn: Callable[Concatenate[TaskContext, P], R], *args: P.args, **kwargs: P.kwargs) -> R:
        new_logger = self.logger.getChild(task_fn.__name__)
        new_context = SimpleTaskContext(new_logger)
        return task_fn(new_context, *args, **kwargs)
