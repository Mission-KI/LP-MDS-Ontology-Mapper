import warnings
from typing import Any

from extended_dataset_profile.models.v0.edp import (
    Resolution,
    VideoDataSet,
    VideoPixelFormat,
)

from edps.taskcontext import TaskContext


async def analyse_video(ctx: TaskContext, format_info: Any, video_stream: Any) -> VideoDataSet:
    codec = video_stream.get("codec_name", "UNKNOWN")
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    fps = _compute_fps(ctx, video_stream.get("avg_frame_rate", "0/1"))
    # Not all formats have the duration in the stream, some only in the format data.
    duration = float(video_stream.get("duration", format_info.get("duration", "0.0")))
    pixel_format = video_stream.get("pix_fmt", "UNKNOWN")

    return VideoDataSet(
        codec=codec,
        resolution=Resolution(width=width, height=height),
        fps=fps,
        duration=duration,
        pixelFormat=VideoPixelFormat(pixel_format),
    )


def _compute_fps(ctx: TaskContext, avg_frame_rate: str) -> float:
    try:
        num, den = avg_frame_rate.split("/")
        return float(num) / float(den)
    except (ValueError, ZeroDivisionError) as error:
        message = f"Could not determine video FPS: {error}"
        ctx.logger.warning(message)
        warnings.warn(message)
        return 0.0
