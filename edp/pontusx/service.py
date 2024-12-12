import re
import shutil
from datetime import datetime
from logging import Logger
from pathlib import Path
from tempfile import TemporaryDirectory

from pydantic import HttpUrl

from edp.compression.zip import ZipAlgorithm
from edp.context import OutputLocalFilesContext
from edp.pontusx.args import Args
from edp.pontusx.metadata import DDO, read_custom_data_file, read_ddo_file
from edp.service import Service
from edp.types import Config, DataSpace, License, Publisher, UserProvidedEdpData

PONTUSX_DS_NAME = "Pontus-X"
PONTUSX_DS_URL = "https://portal.pontus-x.eu"
PONTUSX_ASSET_BASE_URL = f"{PONTUSX_DS_URL}/asset"


def to_UserProvidedEdpData(ddo: DDO) -> UserProvidedEdpData:
    publishDate = ddo.metadata.updated or ddo.metadata.created or datetime.now()

    try:
        license = License(url=str(HttpUrl(ddo.metadata.license)))
    except Exception:
        license = License(name=ddo.metadata.license)

    return UserProvidedEdpData(
        assetId=ddo.id,
        name=ddo.metadata.name,
        description=ddo.metadata.description,
        tags=ddo.metadata.tags,
        url=f"{PONTUSX_ASSET_BASE_URL}/{ddo.id}",
        dataSpace=DataSpace(
            name=PONTUSX_DS_NAME,
            url=PONTUSX_DS_URL,
        ),
        publisher=Publisher(
            name=ddo.metadata.author,
        ),
        publishDate=publishDate,
        license=license,
        freely_available=False,
    )


def sanitize_file_part(file_part: str) -> str:
    # Keep only alphanumeric characters and -_
    return re.sub(r"[^a-zA-Z0-9-_]", "", file_part)


async def run_service(logger: Logger, args: Args):
    custom_data = read_custom_data_file(args.custom_data_file)
    logger.info(f"File extension according to custom data file: {custom_data.fileInfo.fileExtension}")
    ddo = read_ddo_file(args.ddo_file)
    logger.debug(f"DDO: {ddo}")
    user_edp_data = to_UserProvidedEdpData(ddo)
    logger.debug(f"UserProvidedEdpData: {user_edp_data}")

    file_extension = sanitize_file_part(custom_data.fileInfo.fileExtension)
    input_filename = f"data.{file_extension}"
    with (
        TemporaryDirectory() as temp_input_dir_path,
        TemporaryDirectory() as temp_output_dir_path,
    ):
        temp_raw_data_file = Path(temp_input_dir_path) / input_filename
        logger.debug(f"Copying raw data file to {temp_raw_data_file}")
        shutil.copy(args.raw_data_file, temp_raw_data_file)

        edps_config = Config(userProvidedEdpData=user_edp_data)

        output_dir = Path(temp_output_dir_path)
        logger.debug(f"Processing into output dir {output_dir}")
        output_context = OutputLocalFilesContext(output_dir)

        logger.info("Processing asset..")
        edps_service = Service()
        await edps_service.analyse_asset(temp_raw_data_file, edps_config, output_context)

        logger.info("Zipping EDP..")
        target_archive = args.output_dir / f"{sanitize_file_part(user_edp_data.assetId)}.zip"
        await ZipAlgorithm().compress(output_dir, target_archive)
