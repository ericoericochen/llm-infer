from transformers import Qwen3Config

from llminfer.models.qwen3 import Qwen3ForCausalLM


if __name__ == "__main__":
    config = Qwen3Config.from_pretrained("Qwen/Qwen3-0.6B")
    model = Qwen3ForCausalLM(config)
    print(model)
