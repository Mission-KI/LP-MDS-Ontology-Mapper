from zipfile import ZipFile

from pytest import mark, raises

from edps.compression.zip import ZipAlgorithm


@mark.asyncio
async def test_zip_unzip(path_work):
    compression_test_dir = path_work / "compression"
    src_dir = compression_test_dir / "src"
    dest_unzipped = compression_test_dir / "unzipped"
    zip_file = compression_test_dir / "test.zip"

    _create_dummy_files(src_dir)

    zipper = ZipAlgorithm()
    await zipper.compress(src_dir, zip_file)

    with ZipFile(zip_file, "r") as zipf:
        zipf.getinfo("test.txt")
        zipf.getinfo("sub/test2.txt")
        with raises(KeyError):
            zipf.getinfo("missingfile.tmp")

    await zipper.extract(zip_file, dest_unzipped)

    assert (dest_unzipped / "test.txt").exists()
    assert (dest_unzipped / "sub/test2.txt").exists()


def _create_dummy_files(dir):
    (dir / "sub").mkdir(parents=True, exist_ok=True)
    with open(dir / "test.txt", "wt") as text_io_wrapper:
        text_io_wrapper.write("Hello world! test.txt")
    with open(dir / "sub/test2.txt", "wt") as text_io_wrapper:
        text_io_wrapper.write("Hello world! sub/test2.txt")
