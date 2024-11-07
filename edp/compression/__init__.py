from edp.compression.base import CompressionAlgorithm, DecompressionAlgorithm
from edp.compression.zip import ZipAlgorithm

# Dictionary mapping a file extension to CompressionAlgorithm
COMPRESSION_ALGORITHMS: dict[str, CompressionAlgorithm] = {
    "zip": ZipAlgorithm(),
}

# Dictionary mapping a file extension to DecompressionAlgorithm
DECOMPRESSION_ALGORITHMS: dict[str, DecompressionAlgorithm] = {
    "zip": ZipAlgorithm(),
}
