from abc import ABC, abstractmethod
from pathlib import Path


class CompressionAlgorithm(ABC):
    @abstractmethod
    async def extract(self, source_archive: Path, target_directory: Path) -> None: ...
