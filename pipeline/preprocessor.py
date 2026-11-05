import io
from typing import Union
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer
from config import cfg

class CaptionPreprocessor:
    """
    Handles image transformations and token decoding for the ONNX inference engine.
    Decouples raw payload preprocessing from runtime graph execution.
    """
    def __init__(self, encoder_name: str = cfg.encoder_name, decoder_name: str = cfg.decoder_name):
        self.processor = AutoImageProcessor.from_pretrained(encoder_name)
        self.tokenizer = AutoTokenizer.from_pretrained(decoder_name)

    def process_image(self, image_input: Union[str, bytes, Image.Image]) -> np.ndarray:
        """
        Converts file paths, raw bytes, or PIL Images into normalized 
        NumPy arrays ready for ViT ONNX execution.
        """
        if isinstance(image_input, str):
            image = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, bytes):
            image = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
        else:
            raise ValueError("Unsupported input format. Expected path, bytes, or PIL Image.")

        inputs = self.processor(image, return_tensors="np")
        return inputs.pixel_values.astype(np.float32)

    def decode_tokens(self, token_ids: list[int]) -> str:
        """
        Converts generated token IDs back to human-readable string captions.
        """
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @property
    def bos_token_id(self) -> int:
        return self.tokenizer.bos_token_id or 50256

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id or 50256