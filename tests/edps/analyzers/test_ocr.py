from pathlib import Path

import numpy as np
from PIL.Image import open as open_image

from edps.analyzers.images.importer import parse_raster_image
from edps.analyzers.images.ocr import OCR


async def test_ocr_single_line(path_data_test_png, ctx):
    data = load_image_data(path_data_test_png)
    ocr_detector = OCR(languages=["en", "de"])
    detected_texts = ocr_detector.read(data)
    assert len(detected_texts) == 1
    assert detected_texts.iloc[0]["detected text"] == "600 x 400"
    assert detected_texts.iloc[0]["confidence"] > 0.8


async def test_ocr_multi_line(path_data_test_with_text, ctx):
    data = load_image_data(path_data_test_with_text)
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


def load_image_data(path: Path) -> np.ndarray:
    with open_image(path) as img:
        img_metadata, img_array = parse_raster_image(img)
        return img_array
