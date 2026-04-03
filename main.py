import torch
from transformers import AutoTokenizer

from llminfer.models.qwen3 import Qwen3ForCausalLM


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

    messages = [{"role": "user", "content": "Hi, my name is"}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits[0, -1, :]
    probs = logits.softmax(dim=-1)
    next_token_id = torch.multinomial(probs, num_samples=1)
    print(next_token_id)

    next_token = tokenizer.decode(next_token_id[0])
    print(next_token)
