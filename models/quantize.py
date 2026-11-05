import os
from onnxruntime.quantization import quantize_dynamic, QuantType

def compress_to_int8(input_onnx: str, output_onnx: str) -> float:
    """
    Applies INT8 dynamic quantization to weights to optimize for edge CPU execution.
    Returns size reduction percentage.
    """
    quantize_dynamic(
        model_input=input_onnx,
        model_output=output_onnx,
        weight_type=QuantType.QUInt8
    )
    
    size_fp32 = os.path.getsize(input_onnx) / (1024 * 1024)
    size_int8 = os.path.getsize(output_onnx) / (1024 * 1024)
    reduction = ((size_fp32 - size_int8) / size_fp32) * 100
    
    print(f"FP32 Size: {size_fp32:.2f} MB")
    print(f"INT8 Size: {size_int8:.2f} MB")
    print(f"Reduction: {reduction:.1f}%")
    return reduction