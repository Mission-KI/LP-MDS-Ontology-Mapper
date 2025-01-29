from pathlib import PurePosixPath
from typing import AsyncIterator
from uuid import uuid4

import brisque
import cv2
import numpy as np
from extended_dataset_profile.models.v0.edp import ImageColorMode, ImageDataSet, ImageDimensions, ImageDPI
from skimage.exposure import is_low_contrast
from skimage.restoration import estimate_sigma

from edps.analyzers.base import Analyzer
from edps.file import File
from edps.task import TaskContext

brisque_model = brisque.BRISQUE(url=False)


class ImageMetadata:
    def __init__(self, codec: str, color_mode: ImageColorMode, resolution: ImageDimensions, dpi: ImageDPI):
        self.codec = codec
        self.color_mode = color_mode
        self.resolution = resolution
        self.dpi = dpi


class ImageAnalyzer(Analyzer):
    def __init__(self, metadata: ImageMetadata, data: np.ndarray, file: File):
        self._data = data
        self._file = file
        self._metadata = metadata

    async def analyze(self, ctx: TaskContext) -> AsyncIterator[ImageDataSet]:
        height, width, _ = self._data.shape
        ctx.logger.info(
            "Started analysis for image dataset of dimensions %d (width) x %d (height)",
            width,
            height,
        )
        brightness = await self._compute_brightness(self._data)
        blurriness = await self._compute_blurriness(self._data)
        sharpness = await self._compute_sharpness(self._data)
        brisque = await self._compute_brisque(self._data)
        noise = await self._compute_noise(self._data)
        low_contrast = await self._is_low_contrast(self._data)

        yield ImageDataSet(
            uuid=uuid4(),
            parentUuid=None,
            name=PurePosixPath(self._file.relative),
            codec=self._metadata.codec,
            colorMode=self._metadata.color_mode,
            resolution=self._metadata.resolution,
            dpi=self._metadata.dpi,
            brightness=brightness,
            blurriness=blurriness,
            sharpness=sharpness,
            brisque=brisque,
            noise=noise,
            lowContrast=low_contrast,
        )

    async def _compute_brightness(self, img) -> float:
        img_converted = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
        brightness_channel = img_converted[:, :, 1]
        brightness_indication = cv2.mean(brightness_channel)[0]
        return brightness_indication

    async def _compute_blurriness(self, img) -> float:
        return float(cv2.Laplacian(img, cv2.CV_64F).var())

    async def _compute_sharpness(self, img) -> float:
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        gnorm = np.sqrt(laplacian**2)
        sharpness = np.average(gnorm)
        return float(sharpness)

    async def _compute_brisque(self, img) -> float:
        return float(brisque_model.score(img))

    async def _compute_noise(self, img) -> float:
        return float(estimate_sigma(img, channel_axis=-1, average_sigmas=True))

    async def _is_low_contrast(self, img) -> bool:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return bool(is_low_contrast(gray))
