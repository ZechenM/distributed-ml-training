from compression import no_compress, rle_compress, rle_decompress, quantize_lossless_compress, quantize_lossless_decompress


# config 1: define compression method
compression_method = ["no_compress", "rle", "quantization"][1]  # define compression method here by change index
compression_mapping = {
    "no_compress": (no_compress, no_compress),
    "rle": (rle_compress, rle_decompress),
    "quantization": (quantize_lossless_compress, quantize_lossless_decompress),
}

compress, decompress = compression_mapping[compression_method]


# config 2: debug mode
DEBUG = 0