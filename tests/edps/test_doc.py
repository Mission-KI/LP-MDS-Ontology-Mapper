from pydantic_markdown import document_model
from pytest import fixture

from edps.types import Config


@fixture
def output_markdown(path_work):
    with open(path_work / "model_doc.md", "wt", encoding="utf-8") as file:
        yield file


def test_config_doc_is_complete(output_markdown):
    document_model(output_markdown, Config)
