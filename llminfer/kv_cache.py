import torch


class KVCache:
    def __init__(
        self,
        batch_size: int,
        num_layers: int,
        max_seq_len: int,
        num_heads: int,
        head_dim: int,
        device: torch.device | str = "cpu",
    ):
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.cache = [
            (
                torch.zeros(
                    (batch_size, num_heads, max_seq_len, head_dim), device=device
                ),
                torch.zeros(
                    (batch_size, num_heads, max_seq_len, head_dim), device=device
                ),
            )
            for _ in range(num_layers)
        ]

        # one ahead of the last token, starts at 0
        self.pos_idx = 0

    def reset(self):
        self.pos_idx = 0

    def get_cached_layer(self, layer_idx: int):
        k, v = self.get_layer(layer_idx)
        k = k[:, :, : self.pos_idx, :]
        v = v[:, :, : self.pos_idx, :]
        return k, v

    def get_layer(self, layer_idx: int):
        return self.cache[layer_idx]

    def update(self, k: torch.Tensor, v: torch.Tensor, layer_idx: int):
        assert k.shape == v.shape
        kcache, vcache = self.get_layer(layer_idx)
        seq_len = k.shape[2]

        # print("updating k: ", k)

        # store new k and v in cache, handles prefill and decode
        kcache[:, :, self.pos_idx : self.pos_idx + seq_len, :] = k
        vcache[:, :, self.pos_idx : self.pos_idx + seq_len, :] = v

    def advance(self, t: int):
        self.pos_idx += t
