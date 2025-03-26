import json
from typing import Any, AsyncIterator, Dict, Iterable, List, Tuple, Union

from extended_dataset_profile.models.v0.edp import SemiStructuredDataSet
from genson import SchemaBuilder
from pandas import DataFrame, json_normalize

from edps.taskcontext import TaskContext

from ..structured.importer import pandas_importer

JSONData = Union[Dict[str, Any], List[Any]]
StructuredDataMatch = Tuple[Dict[str, Any], str]


class JsonAnalyzer:
    def __init__(self, json_data: JSONData):
        self._json_data = json_data

    async def analyze(self, ctx: TaskContext) -> SemiStructuredDataSet:
        builder = SchemaBuilder()
        builder.add_object(self._json_data)
        schema = builder.to_schema()

        matches = await JsonAnalyzer._detect_structured_data(schema)

        num_dataframes = 0
        async for dataframe in self._extract_dataframes(ctx, matches):
            num_dataframes += 1
            await ctx.exec(f"dataframe_{num_dataframes:03}", pandas_importer, dataframe)

        return SemiStructuredDataSet(
            jsonSchema=json.dumps(schema, indent=2),
        )

    async def _extract_dataframes(
        self, ctx: TaskContext, matches: List[StructuredDataMatch]
    ) -> AsyncIterator[DataFrame]:
        ctx.logger.info("Extracting dataframes...")
        for _, path in matches:
            structured_data = await JsonAnalyzer._extract_structured_data(self._json_data, path)
            dataframe = json_normalize(list(structured_data))
            yield dataframe

    @staticmethod
    async def _detect_structured_data(schema: Dict[str, Any]) -> List[StructuredDataMatch]:
        def traverse(schema_node: Dict[str, Any], path: str):
            json_type = schema_node.get("type")
            if json_type == "object":
                for prop_name, prop_schema in schema_node.get("properties", {}).items():
                    child_path = f"{path}.{prop_name}" if path else prop_name
                    yield from traverse(prop_schema, child_path)
            elif json_type == "array":
                items_schema = schema_node.get("items", {})
                if items_schema.get("type") == "object":
                    yield items_schema, path
                    yield from traverse(items_schema, path)

        matches = list(traverse(schema, ""))
        return matches

    @staticmethod
    async def _extract_structured_data(json_data: JSONData, path: str) -> Iterable[Any]:
        parts = path.split(".") if path else []

        def extract(curr_data: JSONData, curr_parts: List[str]) -> Iterable[Any]:
            if not curr_parts:
                if not (isinstance(curr_data, list) and all(isinstance(item, dict) for item in curr_data)):
                    raise ValueError(
                        f"Invalid path: expected a list of dicts (structured data), but got '{type(curr_data).__name__}'."
                    )
                yield from curr_data
            else:
                if isinstance(curr_data, dict):
                    key, *remaining_parts = curr_parts
                    if key not in curr_data:
                        raise ValueError(f"Invalid path: expected a dict containing key '{key}'.")
                    yield from extract(curr_data[key], remaining_parts)
                elif isinstance(curr_data, list):
                    for item in curr_data:
                        yield from extract(item, curr_parts)
                else:
                    raise ValueError(
                        f"Invalid path: expected a dict or list when processing path '{path}', but got '{type(curr_data).__name__}'."
                    )

        return extract(json_data, parts)
