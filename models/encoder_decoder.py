import torch
import torch.nn as nn
from transformers import VisionEncoderDecoderModel

class ViTCaptioner(nn.Module):
    """
    Encoder-Decoder combining ViT spatial features with GPT-2 generation.
    Uses the pretrained nlpconnect/vit-gpt2-image-captioning checkpoint so the
    cross-attention bridge between image features and tokens is actually trained.
    """
    def __init__(self, encoder_name: str = "nlpconnect/vit-gpt2-image-captioning", decoder_name: str = "gpt2"):
        super().__init__()
        # Load the full pretrained fusion, then keep its trained halves
        full = VisionEncoderDecoderModel.from_pretrained(encoder_name)
        self.encoder = full.encoder
        self.decoder = full.decoder
        
        enc_dim = self.encoder.config.hidden_size
        dec_dim = self.decoder.config.n_embd
        
        self.proj = nn.Linear(enc_dim, dec_dim) if enc_dim != dec_dim else nn.Identity()

    def forward(self, pixel_values: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        # Extract visual tokens
        encoder_out = self.encoder(pixel_values=pixel_values).last_hidden_state
        encoder_out = self.proj(encoder_out)
        
        # Combine visual prefix with sequence tokens
        logits = self.decoder(input_ids=input_ids, encoder_hidden_states=encoder_out).logits
        return logits