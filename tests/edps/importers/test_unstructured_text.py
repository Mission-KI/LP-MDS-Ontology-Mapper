from extended_dataset_profile.models.v0.edp import StructuredDataSet, UnstructuredTextDataSet
from pytest import mark

from edps.importers.unstructured_text import unstructured_text_importer
from edps.taskcontext import TaskContext
from tests.conftest import copy_asset_to_ctx_input_dir


@mark.asyncio
async def test_unstructured_text_only(ctx, path_unstructured_text_only_txt):
    path_in_ctx = copy_asset_to_ctx_input_dir(path_unstructured_text_only_txt, ctx)
    await ctx.exec("dataset", unstructured_text_importer, path_in_ctx)
    datasets = list(ctx.collect_datasets())
    assert len(datasets) == 1
    assert any(isinstance(dataset, UnstructuredTextDataSet) for dataset in datasets)


@mark.asyncio
async def test_unstructured_text_with_table(ctx: TaskContext, path_unstructured_text_with_table):
    path_in_ctx = copy_asset_to_ctx_input_dir(path_unstructured_text_with_table, ctx)
    await ctx.exec("dataset", unstructured_text_importer, path_in_ctx)
    datasets = list(ctx.collect_datasets())
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
    assert unstructured_dataset.lineCount == 2
