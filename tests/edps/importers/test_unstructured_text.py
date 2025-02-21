from typing import List

from extended_dataset_profile.models.v0.edp import StructuredDataSet, UnstructuredTextDataSet
from pytest import mark

from edps.analyzers.base import Analyzer
from edps.analyzers.unstructured_text import Analyzer as UnstructuredTextAnalyzer
from edps.file import File
from edps.importers.unstructured_text import unstructured_text_importer


@mark.asyncio
async def test_unstructured_text_only(ctx, path_unstructured_text_only_txt):
    analyzers: List[Analyzer] = []
    async for analyzer in unstructured_text_importer(
        ctx, File(path_unstructured_text_only_txt.parent, path_unstructured_text_only_txt)
    ):
        analyzers.append(analyzer)
    assert len(analyzers) == 1
    analyzer = analyzers[0]
    assert isinstance(analyzer, UnstructuredTextAnalyzer)


@mark.asyncio
async def test_unstructured_text_with_table(ctx, path_unstructured_text_with_table):
    datasets = [
        dataset
        async for analyzer in unstructured_text_importer(
            ctx, File(path_unstructured_text_with_table.parent, path_unstructured_text_with_table)
        )
        async for dataset in ctx.exec(analyzer.analyze)
    ]
    assert len(datasets) == 2
    assert any(isinstance(dataset, UnstructuredTextDataSet) for dataset in datasets)
    assert any(isinstance(dataset, StructuredDataSet) for dataset in datasets)
    structured_dataset = next(dataset for dataset in datasets if isinstance(dataset, StructuredDataSet))
    assert structured_dataset.columnCount == 3
    assert structured_dataset.rowCount == 2
    headers = [column.name for column in structured_dataset.all_columns]
    assert "id" in headers
    assert "name" in headers
    assert "width" in headers
    assert structured_dataset.all_columns
    assert structured_dataset.numericColumnCount == 2
    assert structured_dataset.stringColumnCount == 1
    assert structured_dataset.datetimeColumnCount == 0
    unstructured_dataset = next(dataset for dataset in datasets if isinstance(dataset, UnstructuredTextDataSet))
    assert len(unstructured_dataset.embeddedTables) == 1
    assert unstructured_dataset.wordCount == 20
    assert unstructured_dataset.lineCount == 4
