from asyncio import get_running_loop
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from edp.compression.base import CompressionAlgorithm, DecompressionAlgorithm


class ZipAlgorithm(CompressionAlgorithm, DecompressionAlgorithm):
    async def compress(self, source_directory: Path, target_archive: Path) -> None:
        await get_running_loop().run_in_executor(None, self._compress, source_directory, target_archive)

    def _compress(self, source_directory: Path, target_archive: Path) -> None:
        # Create a ZipFile object in write mode
        with ZipFile(target_archive, "w", ZIP_DEFLATED) as zip_file:
            # Traverse the folder and its subdirectories
            for root, _, files in source_directory.walk():
                for file in files:
                    # Create the full file path
                    file_path = root / file
                    # Get the relative path to store in the zip file
                    arcname = file_path.relative_to(source_directory)
                    # Write the file to the zip archive
                    zip_file.write(file_path, arcname)

    async def extract(self, source_archive: Path, target_directory: Path) -> None:
        await get_running_loop().run_in_executor(None, self._extract, source_archive, target_directory)

    def _extract(self, source_archive: Path, target_directory: Path) -> None:
        with ZipFile(source_archive, "r") as zip_file:
            zip_file.extractall(target_directory)
