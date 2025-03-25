import json
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic.dataclasses import dataclass
from pydantic_settings import BaseSettings


class RawArgs(BaseSettings):
    """
    These variables are set by Pontus X.
    """

    basedir: Path = Field(description="Data base dir (default: /data)", default=Path("/data"))
    dids: Optional[str] = Field(description='Array of DID identifiers (e.g. ["83274adb"])', default=None)


# Validated and enriched arguments from ENV including derived paths
@dataclass(frozen=True)
class Args:
    did: str
    base_dir: Path
    raw_data_file: Path
    custom_data_file: Path
    ddo_file: Path
    output_dir: Path


def get_args() -> Args:
    raw_args = RawArgs()

    dids_raw = raw_args.dids
    if dids_raw is None:
        raise ValueError("DIDS must be set by ENV variable 'DIDS'")
    try:
        dids_list = json.loads(dids_raw)
        if not isinstance(dids_list, list):
            raise
        if not len(dids_list) == 1:
            raise
        did = dids_list[0]
    except Exception:
        raise TypeError(f"DIDS '{dids_raw}' needs to be a JSON list with exactly one element")

    base_dir = raw_args.basedir.resolve()
    if not base_dir.is_dir():
        raise ValueError(f"Base dir '{base_dir}' is missing. Maybe set ENV variable 'BASEDIR'")

    raw_data_file = base_dir / f"inputs/{did}/0"
    if not raw_data_file.is_file():
        raise ValueError(f"Raw data file '{raw_data_file}' is missing")

    ddo_file = base_dir / f"ddos/{did}"
    if not ddo_file.is_file():
        raise ValueError(f"DDO file '{ddo_file}' is missing")

    output_dir = base_dir / "outputs"
    if not output_dir.is_dir():
        raise ValueError(f"Output dir '{output_dir}' is missing")

    # This file is optional. If it's not found we assume a 'csv' file extension.
    custom_data_file = base_dir / "inputs/algoCustomData.json"

    return Args(
        did=str(did),
        base_dir=base_dir,
        raw_data_file=raw_data_file,
        custom_data_file=custom_data_file,
        ddo_file=ddo_file,
        output_dir=output_dir,
    )
