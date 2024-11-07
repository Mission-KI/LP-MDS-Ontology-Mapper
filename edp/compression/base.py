from abc import ABC, abstractmethod
from pathlib import Path


class CompressionAlgorithm(ABC):
    @abstractmethod
    async def compress(self, source_directory: Path, target_archive: Path) -> None:
        """Compress source_directory and write it to target_archive (should include extension)."""
        ...


class DecompressionAlgorithm(ABC):
    @abstractmethod
    async def extract(self, source_archive: Path, target_directory: Path) -> None:
        """Extract source_archive and write it to target_directory."""
        ...
