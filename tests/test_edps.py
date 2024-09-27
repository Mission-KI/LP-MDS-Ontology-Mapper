from edp import Service, Schema
import pytest
from pathlib import Path

from edp.schema import DataSpace, Schema, DataSetCompression


@pytest.fixture
def edp_schema():
    dataspace_schema = DataSpace(dataSpaceId=123, name="Test Dataspace", url="www.thedataspace.de")
    return Schema(
        assetId=1,
        assetName="Test Asset",
        assetUrl="www.theasset.de",
        assetDataCategory="the_category",
        dataSpace=dataspace_schema,
        publisherId=1234,
        licenseId=12345,
        assetVolume=1000,
        assetVolumeCompressed=500,
        assetCompressionAlgorithm=DataSetCompression.gzip,
    )


def test_load_unknown_dir(edp_schema):
    e = Service(edp_schema)
    with pytest.raises(ValueError):
        e.load_files_in_directory(Path("/does/not/exist/"))


def test_load_pickle_dir(edp_schema):
    e = Service(edp_schema)
    with pytest.raises(ValueError):
        e.load_files_in_directory(Path("./tests/data"))
