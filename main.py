import torch
from transformers import Qwen3Config, AutoTokenizer

from llminfer.models.qwen3 import Qwen3ForCausalLM


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    config = Qwen3Config.from_pretrained("Qwen/Qwen3-0.6B")
    model = Qwen3ForCausalLM(config)

    input_text = "Hi, my name is"
    inputs = tokenizer([input_text, input_text], return_tensors="pt")
    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]

    print(torch.arange(0, 1))
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
