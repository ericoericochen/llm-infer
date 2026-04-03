import torch
from transformers import Qwen3Config

from llminfer.models.qwen3 import Qwen3RotaryEmbeddings


if __name__ == "__main__":
    config = Qwen3Config.from_pretrained("Qwen/Qwen3-0.6B")
    rotary_embeddings = Qwen3RotaryEmbeddings(config)

    position_ids = torch.tensor([[0, 1, 2], [2, 3, 4]])
    sin, cos = rotary_embeddings(position_ids)
    print("sin: ", sin.shape)
    print("cos: ", cos.shape)
