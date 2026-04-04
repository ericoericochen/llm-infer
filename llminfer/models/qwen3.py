import torch
import torch.nn as nn
import torch.nn.functional as F

from safetensors import safe_open
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Iterable
from transformers import Qwen3Config
from huggingface_hub import snapshot_download

from ..kv_cache import KVCache


def create_causal_mask(
    hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    batch_size = hidden_states.shape[0]
    seq_len = hidden_states.shape[1]
    mask = torch.tril(torch.ones((batch_size, seq_len, seq_len))).bool()

    if attention_mask is not None:
        # [batch_size, 1, seq_len] broacastable to mask
        attention_mask = attention_mask.unsqueeze(1).bool()
        mask = mask & attention_mask

    return mask.unsqueeze(1)


def apply_rope(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.cat((y1, y2), dim=-1)


@torch.no_grad()
def load_weights(model: nn.Module, safetensor_files: Iterable[Path | str]):
    for file in safetensor_files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for key in f.keys():
                weight = f.get_tensor(key)
                param = model.get_parameter(key)
                param.copy_(weight)


@dataclass
class CausalLMOutput:
    logits: torch.Tensor
    hidden_states: torch.Tensor


class Qwen3ForCausalLM(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str):
        config = Qwen3Config.from_pretrained(pretrained_model_name_or_path)
        model = cls(config)
        snapshot_dir = snapshot_download(
            pretrained_model_name_or_path, local_files_only=True
        )
        safetensor_files = Path(snapshot_dir).glob("*.safetensors")
        load_weights(model, safetensor_files)
        return model

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ):
        if position_ids is None:
            position_ids = (
                torch.arange(0, input_ids.shape[1], device=self.device)
                .unsqueeze(0)
                .repeat(input_ids.shape[0], 1)
            )

        hidden_states = self.model(
            input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )

        logits = self.lm_head(hidden_states)
        return CausalLMOutput(logits=logits, hidden_states=hidden_states)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_tokens: int = 128,
        temperature: float = 1.0,
    ):
        self.eval()

        # assume batch_size is 1
        assert input_ids.shape[0] == 1
        eos_token_id = self.config.eos_token_id
        for _ in range(max_tokens):
            output = self.forward(input_ids)
            logits = output.logits[:, -1, :]

            if temperature > 0:
                logits = logits / temperature
                probs = logits.softmax(dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)
                token_id = int(next_token_id[0][0])
            else:
                next_token_id = logits.argmax(dim=-1, keepdim=True)
                token_id = int(next_token_id[0][0])

            # autoregressively decode - add next token id back to input_ids
            input_ids = torch.cat((input_ids, next_token_id), dim=1)
            yield token_id
            if token_id == eos_token_id:
                break


class Qwen3Model(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.rope = Qwen3RotaryEmbeddings(config)
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ):
        hidden_states = self.embed_tokens(input_ids)

        # create casual mask
        attention_mask = create_causal_mask(hidden_states, attention_mask)

        # compute rope sin, cos positional embeddings
        positional_embeddings = self.rope(position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                positional_embeddings=positional_embeddings,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


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
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.self_attn = Qwen3Attention(config, layer_idx)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.mlp = Qwen3MLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positional_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.Tensor:
        # pre-norm -> function -> residual

        # self attn
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            positional_embeddings=positional_embeddings,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states

        # mlp
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        rrms = torch.rsqrt(variance + self.eps)
        hidden_states = (rrms * hidden_states) * self.weight
        return hidden_states.to(input_dtype)


class Qwen3Attention(nn.Module):
    """Group Query Attention"""

    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.head_dim = config.head_dim
        self.layer_idx = layer_idx
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
        self.o_proj = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
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
        self,
        hidden_states: torch.Tensor,
        positional_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ):
        batch_size, seq_len, _ = hidden_states.shape

        # project to qkv
        q = self.proj_and_reshape(hidden_states, self.q_proj)
        k = self.proj_and_reshape(hidden_states, self.k_proj)
        v = self.proj_and_reshape(hidden_states, self.v_proj)

        # apply qk-norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # apply rope
        sin, cos = positional_embeddings
        q = apply_rope(q, sin, cos)
        k = apply_rope(k, sin, cos)

        prefill = not kv_cache or (kv_cache and kv_cache.pos_idx == 0)

        # TODO: handle kv cache
        if kv_cache is not None:
            if kv_cache.pos_idx != 0:
                # print("decode: ", self.layer_idx)
                cached_k, cached_v = kv_cache.get_cached_layer(self.layer_idx)
                kv_cache.update(k=k, v=v, layer_idx=self.layer_idx)

                # if self.layer_idx == 0:
                #     print("cached_k: ", cached_k)

                k = torch.cat((cached_k, k), dim=2)
                v = torch.cat((cached_v, v), dim=2)
                # print("k: ", k.shape, "v: ", v.shape, "q: ", q.shape)

                if self.layer_idx == kv_cache.num_layers - 1:
                    kv_cache.advance(seq_len)
            else:
                kv_cache.update(k=k, v=v, layer_idx=self.layer_idx)
                if self.layer_idx == 0:
                    pass
                    # print("going to store k: ", k[0, :, 0, :])
                    # print("going to store v: ", v[0, :, 0, :])
                    # print("cached k is: ", kv_cache.get_layer(0)[0][0, :, 0, :])
                    # print("cached v is: ", kv_cache.get_layer(0)[1][0, :, 0, :])
                if self.layer_idx == kv_cache.num_layers - 1:
                    kv_cache.advance(seq_len)

        if self.layer_idx == 0 and (kv_cache is None or kv_cache.pos_idx != 0):
            pass
            # print("q: ", q[0, :, -1, :])
            # # print("k: ", k[0, :, :, :])
            # # print("v: ", v[0, :, :, :])
            # print("k: ", k.shape)
            # print("v: ", v.shape)

        # group query attention
        group_size = q.shape[1] // k.shape[1]
        k = torch.repeat_interleave(k, repeats=group_size, dim=1)
        v = torch.repeat_interleave(v, repeats=group_size, dim=1)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=prefill)
        # attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # if self.layer_idx == 0 and (kv_cache is None or kv_cache.pos_idx != 0):
        #     print("attn_out: ", attn_out[0, :, -1, :])

        # attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)

        # out projection
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.o_proj(attn_out)

        # if self.layer_idx == 0 and (kv_cache is None or kv_cache.pos_idx != 0):
        #     print("out: ", out[0, -1, :])

        return out


class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config):
        super().__init__()
        hidden_size = config.hidden_size
        intermediate_size = config.intermediate_size
        self.act = nn.SiLU()
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor):
        return self.down_proj(
            self.act(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )
