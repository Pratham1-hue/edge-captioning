import torch
from config import cfg
from models.encoder_decoder import ViTCaptioner
from models.quantize import compress_to_int8

def export():
    model = ViTCaptioner(cfg.encoder_name, cfg.decoder_name)
    model.eval()

    # Dummy inputs for tracing computational graph
    dummy_img = torch.randn(1, 3, cfg.img_size, cfg.img_size)
    dummy_ids = torch.randint(0, 1000, (1, cfg.max_seq_len))

    print("Exporting PyTorch model to ONNX FP32...")
    torch.onnx.export(
        model,
        (dummy_img, dummy_ids),
        cfg.onnx_fp32_path,
        input_names=["pixel_values", "input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "pixel_values": {0: "batch_size"},
            "input_ids": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"}
        },
        opset_version=14
    )
    
    print("Quantizing ONNX graph to INT8...")
    compress_to_int8(cfg.onnx_fp32_path, cfg.onnx_int8_path)

if __name__ == "__main__":
    export()