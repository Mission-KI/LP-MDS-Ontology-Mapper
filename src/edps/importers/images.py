from asyncio import get_running_loop
from typing import AsyncIterator

import numpy as np
from extended_dataset_profile.models.v0.edp import ImageColorMode, ImageDimensions, ImageDPI
from PIL import Image

from edps.analyzers.images import ImageAnalyzer, ImageMetadata
from edps.file import File
from edps.task import TaskContext


async def raster_image_importer(ctx: TaskContext, file: File) -> AsyncIterator[ImageAnalyzer]:
    ctx.logger.info("Importing '%s'", file)

    def runner():
        with Image.open(file.path) as img:
            codec = img.format
            color_mode = img.mode
            resolution = img.size
            dpi = img.info.get("dpi", (0.0, 0.0))
            img_rgb = img.convert(ImageColorMode.RGB)
            img_array = np.array(img_rgb)

        img_metadata = ImageMetadata(
            codec=codec,
            color_mode=ImageColorMode(color_mode),
            resolution=ImageDimensions(width=resolution[0], height=resolution[1]),
            dpi=ImageDPI(x=dpi[0], y=dpi[1]),
        )
        return img_metadata, img_array

    loop = get_running_loop()
    metadata, data = await loop.run_in_executor(None, runner)

    yield ImageAnalyzer(metadata, data, file)
