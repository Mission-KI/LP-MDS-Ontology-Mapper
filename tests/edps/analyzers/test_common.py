from edps.analyzers.common import split_keywords


def test_split_keywords():
    assert split_keywords(None) == []
    assert split_keywords("") == []
    # Ignore one-character keyword strings
    assert split_keywords("x") == []
    assert split_keywords(" x  ") == []
    assert split_keywords("TV") == ["TV"]
    # Strip whitespace characters
    assert split_keywords(" Test  ") == ["Test"]
    # Splitting by ";" has precendence
    assert split_keywords("  Abc;   beta  ;x; gamma ") == ["Abc", "beta", "x", "gamma"]
    # Then splitting by ","
    assert split_keywords("  Abc,   beta  ,x, gamma ") == ["Abc", "beta", "x", "gamma"]
    # Strict priority between separators
    assert split_keywords("  Abc,   beta  ;x, gamma ") == ["Abc,   beta", "x, gamma"]
    # Finally split on whitespace character
    assert split_keywords("  Abc   beta  x gamma ") == ["Abc", "beta", "x", "gamma"]
    assert split_keywords("  Abc\tbeta  x\ngamma ") == ["Abc", "beta", "x", "gamma"]
