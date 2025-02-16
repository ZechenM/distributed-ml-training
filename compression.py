import torch
from typing import Any, Dict, List, Tuple, Set

def rle_compress(data: Dict[str, torch.Tensor]) -> Dict[str, Tuple[List[Tuple[float, int]], Any]]:
    """Compress gradients using Run-Length Encoding (RLE)."""
    # {fc.weight: [(-0.1, 3), (0.2, 2), ...], fc.bias: [(0.0, 5), ...]}
    compressed = {}
    for name, tensor in data.items():
        tensor_flat = tensor.flatten().tolist()  # Flatten and convert to list
        compressed_runs = []
        current_value = tensor_flat[0]
        count = 1
        for value in tensor_flat[1:]:
            if value == current_value:
                count += 1
            else:
                compressed_runs.append((current_value, count))
                current_value = value
                count = 1
        compressed_runs.append((current_value, count))  # Add the last run
        # Store the compressed runs and the original shape
        compressed[name] = (compressed_runs, tensor.shape)
    return compressed


def rle_decompress(
    compressed: Dict[str, Tuple[List[Tuple[float, int]], Any]]
) -> Dict[str, torch.Tensor]:
    """Decompress gradients using Run-Length Encoding (RLE)."""
    decompressed = {}
    for name, (runs, original_shape) in compressed.items():
        tensor_flat = []
        for value, count in runs:
            tensor_flat.extend([value] * count)
        # Reshape to the original shape
        decompressed[name] = torch.tensor(tensor_flat).reshape(original_shape)
    return decompressed
