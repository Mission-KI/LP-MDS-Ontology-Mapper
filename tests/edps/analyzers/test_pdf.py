from typing import Dict

from extended_dataset_profile.models.v0.edp import ModificationState
from pypdf import DocumentInformation
from pypdf.generic import ArrayObject, TextStringObject

from edps.analyzers.pdf import _calc_keywords, _calc_modified, _calc_toolchain


def test_calc_toolchain():
    assert _calc_toolchain(None) is None
    assert _calc_toolchain(DocumentInformation()) is None
    assert _calc_toolchain(_metadata({"/Creator": "Adobe Acrobat"})) == "Adobe Acrobat"
    assert _calc_toolchain(_metadata({"/Creator": "     Adobe Acrobat  "})) == "Adobe Acrobat"
    assert _calc_toolchain(_metadata({"/Producer": "     Adobe Acrobat  "})) == "Adobe Acrobat"
    assert (
        _calc_toolchain(_metadata({"/Creator": "  Microsoft Word  ", "/Producer": "  Adobe Acrobat "}))
        == "Microsoft Word; Adobe Acrobat"
    )
    assert _calc_toolchain(_metadata({"/Creator": "Adobe Acrobat", "/Producer": "  Adobe Acrobat "})) == "Adobe Acrobat"


def test_calc_keywords():
    assert _calc_keywords(None) == []
    assert _calc_keywords("") == []
    # Ignore one-character keyword strings
    assert _calc_keywords("x") == []
    assert _calc_keywords(" x  ") == []
    assert _calc_keywords("TV") == ["TV"]
    # Strip whitespace characters
    assert _calc_keywords(" Test  ") == ["Test"]
    # Splitting by ";" has precendence
    assert _calc_keywords("  Abc;   beta  ;x; gamma ") == ["Abc", "beta", "x", "gamma"]
    # Then splitting by ","
    assert _calc_keywords("  Abc,   beta  ,x, gamma ") == ["Abc", "beta", "x", "gamma"]
    # Strict priority between separators
    assert _calc_keywords("  Abc,   beta  ;x, gamma ") == ["Abc,   beta", "x, gamma"]
    # Finally split on whitespace character
    assert _calc_keywords("  Abc   beta  x gamma ") == ["Abc", "beta", "x", "gamma"]
    assert _calc_keywords("  Abc\tbeta  x\ngamma ") == ["Abc", "beta", "x", "gamma"]


D1 = "D:20250101103000+02'00'"
D2 = "D:20250105123000+02'00'"


def test_calc_modified():
    # ModificationState is calculated primarily depending on the IDs
    assert _calc_modified(None, None) == ModificationState.unknown
    assert _calc_modified(_ids("1321", "1321"), None) == ModificationState.unmodified
    assert _calc_modified(_ids("1321", "1355"), None) == ModificationState.modified
    # Set state to modified if dates differ
    assert _calc_modified(None, _metadata({"/CreationDate": D1, "/ModDate": D2})) == ModificationState.modified
    assert (
        _calc_modified(_ids("1321", "1321"), _metadata({"/CreationDate": D1, "/ModDate": D2}))
        == ModificationState.modified
    )
    # But never set it to unmodified if dates are equal
    assert _calc_modified(None, _metadata({"/CreationDate": D1, "/ModDate": D1})) == ModificationState.unknown
    assert (
        _calc_modified(_ids("1321", "1355"), _metadata({"/CreationDate": D1, "/ModDate": D1}))
        == ModificationState.modified
    )


def _metadata(data: Dict[str, str]) -> DocumentInformation:
    di = DocumentInformation()
    for key, value in data.items():
        di[TextStringObject(key)] = TextStringObject(value)
    return di


def _ids(initial_id: str, current_id: str) -> ArrayObject:
    ids = ArrayObject()
    ids.append(TextStringObject(initial_id))
    ids.append(TextStringObject(current_id))
    return ids
