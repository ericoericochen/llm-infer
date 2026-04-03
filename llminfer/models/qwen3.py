import torch
import torch.nn as nn

from transformers import Qwen3Config


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)

    @torch.inference_mode()
    def generate(self):
        pass


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(config.hidden_size)
        self.self_attn = Qwen3Attention()
        self.post_attention_layernorm = nn.LayerNorm(config.hidden_size)
        self.mlp = Qwen3MLP()


class Qwen3Attention(nn.Module):
    pass


class Qwen3MLP(nn.Module):
    pass
