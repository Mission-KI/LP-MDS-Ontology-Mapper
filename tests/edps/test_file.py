from pathlib import Path

from pytest import raises

from edps.file import build_real_sub_path


def test_build_real_sub_path():
    base = Path("/usr/edps/work")
    assert build_real_sub_path(base, "dir/test.html").as_posix() == "/usr/edps/work/dir/test.html"
    assert build_real_sub_path(base, Path("dir/test.html")).as_posix() == "/usr/edps/work/dir/test.html"
    assert (
        build_real_sub_path(base, Path("dir/../../work/dir/test.html")).resolve()
        == Path("/usr/edps/work/dir/test.html").resolve()
    )
    assert (
        build_real_sub_path(base, Path("/usr/edps/work/dir/test.html")).resolve()
        == Path("/usr/edps/work/dir/test.html").resolve()
    )
    with raises(ValueError):
        build_real_sub_path(base, "dir/../../test.html")
    with raises(ValueError):
        build_real_sub_path(base, "/test.html")

    base = Path("./work")
    assert build_real_sub_path(base, "dir/test.html").as_posix() == "work/dir/test.html"
    with raises(ValueError):
        build_real_sub_path(base, "dir/../../test.html")
    with raises(ValueError):
        build_real_sub_path(base, "/test.html")
