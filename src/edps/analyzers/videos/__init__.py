from pathlib import PurePosixPath
from uuid import uuid4

from extended_dataset_profile.models.v0.edp import (
    Resolution,
    VideoCodec,
    VideoDataSet,
    VideoPixelFormat,
)
from pydantic.dataclasses import dataclass

from edps.taskcontext import TaskContext


@dataclass(frozen=True)
class VideoMetadata:
    codec: VideoCodec
    resolution: Resolution
    fps: float
    duration: float
    pixel_format: VideoPixelFormat


class VideoAnalyzer:
    def __init__(self, metadata: VideoMetadata):
        self._metadata = metadata

    async def analyze(self, ctx: TaskContext) -> VideoDataSet:
        ctx.logger.info("Started analysis for video dataset")

        return VideoDataSet(
            uuid=uuid4(),  # TODO uuid, parentUuid & name are set by the TaskContext and don't need explicit initialization!
            parentUuid=None,
            name=PurePosixPath(""),
            codec=self._metadata.codec,
            resolution=self._metadata.resolution,
            fps=self._metadata.fps,
            duration=self._metadata.duration,
            pixel_format=self._metadata.pixel_format,
        )
