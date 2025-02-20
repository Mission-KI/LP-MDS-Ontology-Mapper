from abc import ABC, abstractmethod
from typing import Iterator


class ChunkInterface(ABC):
    @property
    @abstractmethod
    def start_line_inclusive(self) -> int: ...

    @property
    @abstractmethod
    def end_line_exclusive(self) -> int: ...

    def __contains__(self, line_index: int) -> bool:
        return self.start_line_inclusive <= line_index < self.end_line_exclusive

    def overlaps(self, other: "ChunkInterface") -> bool:
        return (other.start_line_inclusive < self.end_line_exclusive) and (
            self.start_line_inclusive < other.end_line_exclusive
        )


class Chunk(ChunkInterface):
    def __init__(self, start_position_inclusive: int, end_position_exclusive: int):
        if end_position_exclusive < start_position_inclusive:
            raise RuntimeError("A chunk has to be constructed with a start BEFORE the end position.")
        self._start_position_inclusive = start_position_inclusive
        self._end_position_exclusive = end_position_exclusive

    @property
    def start_line_inclusive(self) -> int:
        return self._start_position_inclusive

    @start_line_inclusive.setter
    def start_line_inclusive(self, value: int) -> None:
        if value > self._end_position_exclusive:
            raise RuntimeError(
                "Can not set start position of chunk to a value larger than the end position! "
                "You may want to create a new Chunk object for that."
            )
        self._start_position_inclusive = value

    @property
    def end_line_exclusive(self) -> int:
        return self._end_position_exclusive

    @end_line_exclusive.setter
    def end_line_exclusive(self, value: int) -> None:
        if value < self._start_position_inclusive:
            raise RuntimeError(
                "Can not set end position to a value smaller than the start position! "
                "You may want to create a new Chunk object for that."
            )
        self._end_position_exclusive = value

    def __sub__(self, other: "ChunkInterface") -> Iterator["Chunk"]:
        if not self.overlaps(other):
            yield Chunk(
                start_position_inclusive=self.start_line_inclusive,
                end_position_exclusive=self.end_line_exclusive,
            )
        elif (other.start_line_inclusive <= self.start_line_inclusive) and (
            other.end_line_exclusive >= self.end_line_exclusive
        ):
            # Other encompasses self.
            pass
        elif (other.start_line_inclusive > self.start_line_inclusive) and (
            other.end_line_exclusive < self.end_line_exclusive
        ):
            # Other is inside self, split self in two.
            yield Chunk(
                start_position_inclusive=self.start_line_inclusive,
                end_position_exclusive=other.start_line_inclusive,
            )
            yield Chunk(
                start_position_inclusive=other.end_line_exclusive,
                end_position_exclusive=self.end_line_exclusive,
            )
        else:
            # The subtract self is either at the beginning or end of self.
            if other.start_line_inclusive <= self.start_line_inclusive:
                start_position = other.end_line_exclusive
            else:
                start_position = self.start_line_inclusive

            if other.end_line_exclusive >= self.end_line_exclusive:
                end_position = other.start_line_inclusive
            else:
                end_position = self.end_line_exclusive

            yield Chunk(start_position_inclusive=start_position, end_position_exclusive=end_position)
