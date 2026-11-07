import os
from dataclasses import dataclass

@dataclass
class Config:
    encoder_name: str = "nlpconnect/vit-gpt2-image-captioning"
    decoder_name: str = "gpt2"
    max_seq_len: int = 32
    img_size: int = 224
    
    # Export paths
    onnx_fp32_path: str = "weights/model_fp32.onnx"
    onnx_int8_path: str = "weights/model_int8.onnx"

cfg = Config()
os.makedirs("weights", exist_ok=True)