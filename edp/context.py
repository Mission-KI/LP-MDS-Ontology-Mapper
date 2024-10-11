from abc import ABC, abstractmethod
from asyncio import get_running_loop
from contextlib import asynccontextmanager
from io import TextIOWrapper
from json import load as json_load
from logging import getLogger
from pathlib import Path, PurePosixPath
from typing import AsyncIterator, Tuple
from uuid import UUID, uuid4

from boto3 import resource as aws_resource
from matplotlib import colormaps, use
from matplotlib.axes import Axes
from matplotlib.colors import Colormap, LinearSegmentedColormap
from matplotlib.pyplot import close as close_figure
from matplotlib.pyplot import set_cmap, subplots
from matplotlib.style import use as use_style
from pydantic import HttpUrl
from requests import post
from seaborn import reset_orig

from edp.types import FileReference

DEFAULT_STYLE_PATH = Path(__file__).parent / "styles/plot.mplstyle"


class TextWriter(ABC):
    """Abstract class for object that supports async writes with text"""

    @abstractmethod
    async def write(self, text: str) -> None: ...


class TextFileWriter(TextWriter):
    def __init__(self, io_wrapper: TextIOWrapper) -> None:
        self._wrapper = io_wrapper

    async def write(self, text: str) -> None:
        loop = get_running_loop()
        await loop.run_in_executor(None, self._wrapper.write, text)


class OutputContext(ABC):
    """Abstract class that provides functions to generate files for the service and reference them.

    Depending on the implementation, these files will be stored locally or in the cloud."""

    @abstractmethod
    @asynccontextmanager
    def get_text_file(self, name: str) -> AsyncIterator[Tuple[TextWriter, FileReference]]: ...

    @abstractmethod
    @asynccontextmanager
    def get_plot(self, name: str) -> AsyncIterator[Tuple[Axes, FileReference]]: ...


def _get_default_colormap() -> Colormap:
    BLUE = "#43ACFF"
    PINK = "#FF3FFF"
    # GRAY = "#D9D9D9"
    colormap = LinearSegmentedColormap.from_list("daseen", [BLUE, PINK])
    colormaps.register(colormap)
    return colormap


class OutputLocalFilesContext(OutputContext):
    """This supplies functions to generate output files and graphs."""

    def __init__(
        self,
        path: Path,
        text_encoding: str = "utf-8",
        matplotlib_backend: str = "AGG",
        default_plot_format: str = "png",
        colormap: Colormap = _get_default_colormap(),
    ) -> None:
        self._logger = getLogger(__name__)
        if path.exists() and not path.is_dir():
            raise RuntimeError(f'Output path "{path}" must be a directory!')
        if not path.exists():
            self._logger.info('Creating output path "%s"', path)
            path.mkdir(parents=True)
        self.path = path
        self.text_encoding = text_encoding
        use(matplotlib_backend)
        self._default_plot_format = default_plot_format
        reset_orig()
        use_style(DEFAULT_STYLE_PATH)
        set_cmap(colormap)

    def build_full_path(self, relative_path: PurePosixPath):
        return PurePosixPath(self.path / relative_path)

    @asynccontextmanager
    async def get_text_file(self, name: str):
        save_path = self._prepare_save_path(name, ".json")
        with open(save_path, "wt", encoding=self.text_encoding) as io_wrapper:
            yield TextFileWriter(io_wrapper), PurePosixPath(save_path.relative_to(self.path))
        self._logger.debug('Generated text file "%s"', save_path)

    @asynccontextmanager
    async def get_plot(self, name: str):
        save_path = self._prepare_save_path(name, "." + self._default_plot_format)
        figure, axes = subplots()
        yield axes, PurePosixPath(save_path.relative_to(self.path))
        figure.savefig(save_path)
        close_figure(figure)
        self._logger.debug('Generated plot "%s"', save_path)

    def _prepare_save_path(self, name: str, suffix: str):
        save_path = self.path / (name.replace(":", "").replace(" ", "") + suffix)
        if save_path.exists():
            self._logger.warning('The path "%s" already exists, will overwrite!', save_path)
            save_path.unlink()
        else:
            save_path.parent.mkdir(parents=True, exist_ok=True)
        return save_path


class OutputDaseenContext(OutputContext):
    """Upload images to AWS S3 and JSON to Elastic Search"""

    def __init__(
        self,
        local_path: Path,
        s3_access_key_id: str,
        s3_secret_access_key: str,
        s3_bucket_name: str,
        elastic_url: str,
        elastic_apikey: str,
        text_encoding: str = "utf-8",
        request_timeout: float = 10,
    ):
        self._logger = getLogger(__name__)
        self.output_local_context = OutputLocalFilesContext(local_path, text_encoding)
        self.s3_bucket = self._build_s3_bucket(s3_access_key_id, s3_secret_access_key, s3_bucket_name)
        self.elastic_url = elastic_url
        self.elastic_apikey = elastic_apikey
        self.request_timeout = request_timeout

    @asynccontextmanager
    async def get_text_file(self, name: str):
        async with self.output_local_context.get_text_file(name) as (textFileWrite, rel_path):
            docid: UUID = uuid4()
            download_url = self._build_elastic_download_url(docid)
            yield textFileWrite, download_url
        self._upload_to_elastic(rel_path, docid, download_url)

    @asynccontextmanager
    async def get_plot(self, name: str):
        async with self.output_local_context.get_plot(name) as (axes, rel_path):
            upload_key = self._build_s3_key(rel_path)
            download_url = self._build_s3_download_url(upload_key)
            yield axes, download_url
        self._upload_to_s3(rel_path, upload_key, download_url)

    def _build_s3_bucket(self, s3_access_key_id: str, s3_secret_access_key: str, s3_bucket_name: str):
        s3 = aws_resource("s3", aws_access_key_id=s3_access_key_id, aws_secret_access_key=s3_secret_access_key)
        return s3.Bucket(s3_bucket_name)

    def _build_s3_key(self, file_ref: PurePosixPath):
        return f"{uuid4()}/{file_ref.name}"

    def _build_s3_download_url(self, upload_key: str) -> FileReference:
        return HttpUrl(f"https://{self.s3_bucket.name}.s3.amazonaws.com/{upload_key}")

    def _upload_to_s3(self, file_rel_path: PurePosixPath, upload_key: str, download_url: FileReference):
        file_full_path = self.output_local_context.build_full_path(file_rel_path)
        self.s3_bucket.upload_file(file_full_path, upload_key)
        self._logger.info(f"Uploaded {file_full_path} to S3: {download_url}")

    def _build_elastic_download_url(self, docid: UUID) -> FileReference:
        return HttpUrl(f"{self.elastic_url}/_doc/{docid}")

    def _upload_to_elastic(self, file_rel_path: PurePosixPath, docid: UUID, download_url: FileReference):
        file_full_path = self.output_local_context.build_full_path(file_rel_path)
        url = f"{self.elastic_url}/_create/{docid}"
        headers = {"Authorization": f"ApiKey {self.elastic_apikey}"}
        # We expect a JSON file, so Elastic Search can handle it.
        self._logger.info(f"Loading file {file_full_path} for insertion into Elastic")
        with open(file_full_path, "r") as file:
            json_data = json_load(file)
        response = post(url=url, json=json_data, headers=headers, timeout=self.request_timeout)
        response.raise_for_status()
        self._logger.info(f"Uploaded {file_full_path} to Elastic Search with ID {docid}: {download_url}")
