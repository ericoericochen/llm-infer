import torch
from transformers import AutoTokenizer

from llminfer.models.qwen3 import Qwen3ForCausalLM
from llminfer.kv_cache import KVCache


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    model = Qwen3ForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
    model.eval()
    kv_cache = KVCache(
        batch_size=1,
        num_layers=model.config.num_hidden_layers,
        max_seq_len=512,
        num_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
    )

    device = model.device

    # prepare input message
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

    kv_cache.reset()
    max_tokens = 128
    with torch.inference_mode():
        # prefill - compute k,v for entire input tokens
        output = model(
            input_ids=input_ids, attention_mask=attention_mask, kv_cache=kv_cache
        )

        input_seq_len = input_ids.shape[1]
        next_position_id = torch.tensor([[input_seq_len]])
        logits = output.logits[:, -1, :]

        for i in range(max_tokens):
            next_token_id = logits.argmax(dim=-1, keepdim=True)
            next_token = tokenizer.decode(next_token_id)[0]

            print(next_token, flush=True, end="")

            if int(next_token_id[0][0]) == tokenizer.eos_token_id:
                break

            if i == max_tokens - 1:
                continue

            # print("next_token_id: ", next_token_id, flush=True)
            # print("next_position_id: ", next_position_id, flush=True)

            output = model(
                input_ids=next_token_id,
                position_ids=next_position_id,
                kv_cache=kv_cache,
            )
            logits = output.logits[:, -1, :]

            # update next position id
            next_position_id.add_(1)
