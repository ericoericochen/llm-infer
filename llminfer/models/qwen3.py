import torch
import torch.nn as nn

from typing import Optional
from transformers import Qwen3Config


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ):
        hidden_states = self.model(input_ids, attention_mask)

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
        self.rope = Qwen3RotaryEmbeddings(config)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ):
        hidden_states = self.embed_tokens(input_ids)

        # create casual mask

        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask=attention_mask)
            break


def apply_rope(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    pass


class Qwen3RotaryEmbeddings(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        head_dim = config.head_dim
        rope_theta = config.rope_parameters["rope_theta"]
        inv_freqs = rope_theta ** (-2 * torch.arange(0, head_dim // 2) / head_dim)
        positions = torch.arange(0, config.max_position_embeddings)
        freqs = torch.outer(positions, inv_freqs)
        sin, cos = torch.sin(freqs), torch.cos(freqs)
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

    def forward(self, position_ids: torch.Tensor):
        # broadcastable to [batch_size, num_heads, seq_len, head_dim]
        sin = self.sin[position_ids].unsqueeze(1)
        cos = self.cos[position_ids].unsqueeze(1)
        return sin, cos


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen3Attention(config)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = Qwen3MLP()

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, attention_mask)

        # print("hidden_states: ", hidden_states.shape)


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).sum(-1, keepdim=True)
        rrms = torch.rsqrt(variance + self.eps)
        hidden_states = (rrms * hidden_states) * self.weight
        return hidden_states.to(input_dtype)


def create_causal_mask(
    hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    pass


class Qwen3Attention(nn.Module):
    """Group Query Attention"""

    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            bias=config.attention_bias,
        )

        self.q_norm = Qwen3RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(config.head_dim, eps=config.rms_norm_eps)

    def proj_and_reshape(self, hidden_states: torch.Tensor, proj: nn.Linear):
        batch_size, seq_len, _ = hidden_states.shape
        # [batch_size, num_heads, seq_len, head_dim]
        return (
            proj(hidden_states)
            .view(batch_size, seq_len, -1, self.head_dim)
            .transpose(1, 2)
        )

    def forward(
        self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
    ):
        # project to qkv
        q = self.proj_and_reshape(hidden_states, self.q_proj)
        k = self.proj_and_reshape(hidden_states, self.k_proj)
        v = self.proj_and_reshape(hidden_states, self.v_proj)

        # apply qk-norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # apply rope

        # self attention


class Qwen3MLP(nn.Module):
    pass
