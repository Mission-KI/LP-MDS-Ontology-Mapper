from asyncio import get_running_loop
from pathlib import PurePosixPath
from typing import AsyncIterator

import numpy as np
from extended_dataset_profile.models.v0.edp import ImageColorMode, ImageDimensions, ImageDPI
from PIL.Image import Image
from PIL.Image import open as open_image

from edps.analyzers.images import ImageAnalyzer, ImageMetadata
from edps.file import File
from edps.task import TaskContext


async def raster_image_importer(ctx: TaskContext, file: File) -> AsyncIterator[ImageAnalyzer]:
    ctx.logger.info("Importing '%s'", file)

    def runner():
        with open_image(file.path) as img:
            return build_raster_image_analyzer(img, PurePosixPath(file.relative))

    loop = get_running_loop()
    analyzer = await loop.run_in_executor(None, runner)
    yield analyzer


def build_raster_image_analyzer(img: Image, name: PurePosixPath) -> ImageAnalyzer:
    codec = img.format
    color_mode = img.mode
    resolution = img.size
    dpi = img.info.get("dpi", (0.0, 0.0))
    img_rgb = img.convert(ImageColorMode.RGB)
    img_array = np.array(img_rgb)

    img_metadata = ImageMetadata(
        codec=codec or "UNKNOWN",
        color_mode=ImageColorMode(color_mode),
        resolution=ImageDimensions(width=resolution[0], height=resolution[1]),
        dpi=ImageDPI(x=dpi[0], y=dpi[1]),
    )
    return ImageAnalyzer(img_metadata, img_array, name)
