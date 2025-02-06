from pathlib import Path

from edps.analyzers.images.ocr import OCR
from edps.file import File
from edps.importers.images import raster_image_importer


async def test_ocr_single_line(path_data_test_png, download_ocr_models, ctx):
    file = get_file(path_data_test_png)
    analyzer = await anext(ctx.exec(raster_image_importer, file))
    data = analyzer._data
    ocr_detector = OCR(languages=["en", "de"])
    detected_texts = ocr_detector.read(data)
    assert len(detected_texts) == 1
    assert detected_texts.iloc[0]["detected text"] == "600 x 400"
    assert detected_texts.iloc[0]["confidence"] > 0.8


async def test_ocr_multi_line(path_data_test_with_text, download_ocr_models, ctx):
    file = get_file(path_data_test_with_text)
    analyzer = await anext(ctx.exec(raster_image_importer, file))
    data = analyzer._data
    ocr_detector = OCR(languages=["en", "de"])
    detected_texts = ocr_detector.read(data)
    assert len(detected_texts) == 4
    assert detected_texts.iloc[0]["detected text"] == "Hello"
    assert detected_texts.iloc[0]["confidence"] > 0.9
    assert detected_texts.iloc[1]["detected text"] == "World 777 XYZ"
    assert detected_texts.iloc[1]["confidence"] > 0.9
    assert detected_texts.iloc[2]["detected text"] == "TEST ABC"
    assert detected_texts.iloc[2]["confidence"] > 0.9
    assert detected_texts.iloc[3]["detected text"] == "123456789"
    assert detected_texts.iloc[3]["confidence"] > 0.9


def get_file(path: Path):
    return File(path.parent, path)
