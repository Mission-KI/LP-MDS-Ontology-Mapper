import os
import zipfile
from pathlib import Path


def zip_directory(folder_path: Path, output_path: Path):
    """Zip folder and write it to output_path (should normally have "zip" extension)."""

    # Create a ZipFile object in write mode
    with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        # Traverse the folder and its subdirectories
        for root, _, files in os.walk(folder_path):
            for file in files:
                # Create the full file path
                file_path = os.path.join(root, file)
                # Get the relative path to store in the zip file
                arcname = os.path.relpath(file_path, folder_path)
                # Write the file to the zip archive
                zipf.write(file_path, arcname)
