import json
import os
from argparse import ArgumentParser
from pathlib import Path
from typing import Optional

from pydantic.dataclasses import dataclass


# Arguments from ENV / command-line including derived paths
@dataclass(frozen=True)
class Args:
    did: str
    base_dir: Path
    raw_data_file: Path
    custom_data_file: Path
    ddo_file: Path
    output_dir: Path


def parse_args(cmd_args: list[str]) -> Args:
    parser = ArgumentParser()
    parser.add_argument("--basedir", type=Path, default=Path("/data"), help="Data base dir (default: /data)")
    parser.add_argument("--dids", type=str, help="Array of DID identifiers (default: ENV var DIDS)")
    raw_args = parser.parse_args(cmd_args)

    dids_raw: Optional[str] = raw_args.dids
    if dids_raw is None:
        dids_raw = os.getenv("DIDS")
        if dids_raw is None:
            raise ValueError("DIDS must be set either by ENV variable 'DIDS' or cmd-line '--dids'")
    try:
        dids_list = json.loads(dids_raw)
        if not isinstance(dids_list, list):
            raise
        if not len(dids_list) == 1:
            raise
        did = dids_list[0]
    except Exception:
        raise TypeError(f"DIDS '{dids_raw}' needs to be a JSON list with exactly one element")

    base_dir = Path(raw_args.basedir).resolve()
    if not base_dir.is_dir():
        raise ValueError(f"Base dir '{base_dir}' is missing. Maybe set '--basedir'")

    raw_data_file = base_dir / f"inputs/{did}/0"
    if not raw_data_file.is_file():
        raise ValueError(f"Raw data file '{raw_data_file}' is missing")

    custom_data_file = base_dir / "inputs/algoCustomData.json"
    if not custom_data_file.is_file():
        raise ValueError(f"Custom data file '{custom_data_file}' is missing")

    ddo_file = base_dir / f"ddos/{did}"
    if not ddo_file.is_file():
        raise ValueError(f"DDO file '{ddo_file}' is missing")

    output_dir = base_dir / "outputs"
    if not output_dir.is_dir():
        raise ValueError(f"Output dir '{output_dir}' is missing")

    return Args(
        did=str(did),
        base_dir=base_dir,
        raw_data_file=raw_data_file,
        custom_data_file=custom_data_file,
        ddo_file=ddo_file,
        output_dir=output_dir,
    )
