import shutil
from logging import getLogger
from pathlib import Path
from tempfile import TemporaryDirectory

from edp.pontusx.args import parse_args
from edp.pontusx.service import run_service

# PYTHON_BIN: str = sys.executable
EDPS_DIR = (Path(__file__).parent / "../..").resolve()
TEST_DATA_DIR = EDPS_DIR / "tests/data"
DID = "did230948"


_logger = getLogger(__file__)


async def test_cli():
    with TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        _logger.info(f"Preparing Pontux-X container data dir: {temp_dir}")
        # Create dirs
        (temp_dir_path / "ddos").mkdir()
        (temp_dir_path / f"inputs/{DID}").mkdir(parents=True)
        (temp_dir_path / "outputs").mkdir()
        # Copy DDO to DATA/ddos/DID (without file-extension)
        shutil.copy(TEST_DATA_DIR / "pontusx/ddo.json", temp_dir_path / f"ddos/{DID}")
        # Copy algoCustomData.json to DATA/inputs/algoCustomData.json
        shutil.copy(
            TEST_DATA_DIR / "pontusx/algoCustomData.json",
            temp_dir_path / "inputs/algoCustomData.json",
        )
        # Copy test.csv to DATA/inputs/0 (without file-extension)
        shutil.copy(
            TEST_DATA_DIR / "test.csv",
            temp_dir_path / f"inputs/{DID}/0",
        )

        args = parse_args([f"--basedir={temp_dir}", f'--dids=["{DID}"]'])
        await run_service(_logger, args)

        # Expect a ZIP file
        files = [file.name for file in (temp_dir_path / "outputs").iterdir() if file.is_file()]
        _logger.info(f"Got output files: {files}")
        assert len(files) == 1
        assert files[0].endswith(".zip")
