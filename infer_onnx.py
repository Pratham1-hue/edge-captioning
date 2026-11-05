from typing import Union
import numpy as np
import onnxruntime as ort
from PIL import Image
from config import cfg
from pipeline.preprocessor import CaptionPreprocessor

class EdgeCaptionEngine:
    def __init__(self, model_path: str = cfg.onnx_int8_path):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.preprocessor = CaptionPreprocessor()

    def generate_caption(self, image_input: Union[str, bytes, Image.Image]) -> str:
        pixel_values = self.preprocessor.process_image(image_input)
        tokens = [self.preprocessor.bos_token_id]
        
        for _ in range(cfg.max_seq_len):
            input_ids = np.array([tokens], dtype=np.int64)
            inputs = {
                "pixel_values": pixel_values,
                "input_ids": input_ids
            }
            
            logits = self.session.run(None, inputs)[0]
            next_token = int(np.argmax(logits[0, -1, :]))
            
            if next_token == self.preprocessor.eos_token_id:
                break
            tokens.append(next_token)
            
        return self.preprocessor.decode_tokens(tokens)