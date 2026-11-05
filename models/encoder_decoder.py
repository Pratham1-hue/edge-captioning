import torch
import torch.nn as nn
from transformers import ViTModel, GPT2LMHeadModel

class ViTCaptioner(nn.Module):
    """
    Encoder-Decoder combining ViT spatial features with GPT-2 generation.
    Projection layer aligns ViT patch dim (768) with GPT-2 embed dim (768).
    """
    def __init__(self, encoder_name: str = "google/vit-base-patch16-224-in21k", decoder_name: str = "gpt2"):
        super().__init__()
        self.encoder = ViTModel.from_pretrained(encoder_name)
        self.decoder = GPT2LMHeadModel.from_pretrained(decoder_name)
        
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