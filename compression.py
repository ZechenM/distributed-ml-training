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


def quantize_lossless_compress(gradients: dict, num_bits=8):
    """
    Quantizes the gradient tensors using uniform quantization (loseless approach).

    Parameters:
    gradients (dict): Dictionary of named PyTorch gradient tensors.
    num_bits (int): Number of bits for quantization (default: 8-bit).

    Returns:
    dict: Dictionary containing quantized values and metadata for reconstruction.
    """
    quantized_data = {}
    levels = 2**num_bits - 1

    for name, grad in gradients.items():
        min_val, max_val = grad.min(), grad.max()
        quantized_values = torch.round(
            (grad - min_val) / (max_val - min_val) * levels
        ).to(torch.uint8)

        quantized_data[name] = {
            "min_val": min_val.item(),
            "max_val": max_val.item(),
            "quantized_values": quantized_values.cpu().numpy().tolist(),
            "num_bits": num_bits,
        }

    return quantized_data


def quantize_lossless_decompress(q_data: dict):
    """
    Decompresses the quantized gradient data back to PyTorch tensors.

    Parameters:
    q_data (dict): Dictionary of quantized gradient data from quantize_gradients().

    Returns:
    dict: Dictionary of decompressed PyTorch tensors.
    """
    decompressed_data = {}

    for name, data in q_data.items():
        levels = 2 ** data["num_bits"] - 1
        quantized_values = torch.tensor(data["quantized_values"], dtype=torch.float32)
        decompressed_values = data["min_val"] + (quantized_values / levels) * (
            data["max_val"] - data["min_val"]
        )
        decompressed_data[name] = decompressed_values

    return decompressed_data
