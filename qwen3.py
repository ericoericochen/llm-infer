from transformers import AutoTokenizer

from llminfer.models.qwen3 import Qwen3ForCausalLM


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

    device = model.device
    messages = [
        {
            "role": "user",
            "content": "What is the pH scale in one sentence?",
        }
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tokenizer([text], return_tensors="pt").to(device)
    input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]

    for token_id in model.generate(input_ids, temperature=0.0):
        token = tokenizer.decode(token_id)
        print(token, end="", flush=True)
