import os
from onnxruntime.quantization import quantize_dynamic, QuantType

def _on_disk_size(path: str) -> float:
    """
    Total size of a model on disk. Dynamo-exported ONNX graphs may store
    weights in an external <name>.onnx.data file, so sum both when present.
    """
    total = os.path.getsize(path)
    external = path + ".data"
    if os.path.exists(external):
        total += os.path.getsize(external)
    return total / (1024 * 1024)

def compress_to_int8(input_onnx: str, output_onnx: str) -> float:
    """
    Applies INT8 dynamic quantization to weights to optimize for edge CPU execution.
    Returns size reduction percentage, counting external data files.
    """
    quantize_dynamic(
        model_input=input_onnx,
        model_output=output_onnx,
        weight_type=QuantType.QUInt8
    )

    size_fp32 = _on_disk_size(input_onnx)
    size_int8 = _on_disk_size(output_onnx)
    reduction = ((size_fp32 - size_int8) / size_fp32) * 100

    print(f"FP32 Size: {size_fp32:.2f} MB")
    print(f"INT8 Size: {size_int8:.2f} MB")
    print(f"Reduction: {reduction:.1f}%")
    return reduction
