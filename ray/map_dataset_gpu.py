"""
Stateful transform: https://docs.ray.io/en/latest/data/transforming-data.html#stateful-transforms
"""

from __future__ import annotations

import json

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

import ray
import ray.data


class Generator:
    def __init__(self, model_id: str) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left", trust_remote_code=True)
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_id,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
            device_map="auto",
            use_cache=True,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        conversations = [json.loads(conv) for conv in batch["conversation"]]
        inputs = self.tokenizer.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        ).to("cuda")

        sequence_ids = self.model.generate(**inputs, do_sample=False, max_new_tokens=256)

        output_ids = sequence_ids[:, inputs["input_ids"].shape[1] :].cpu()
        output_texts = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        for conv, output_text in zip(conversations, output_texts):
            conv.append({"role": "assistant", "content": output_text})
        batch["conversation"] = [json.dumps(conv) for conv in conversations]
        return batch


model_id = "Qwen/Qwen2.5-0.5B-Instruct"
batch_size = 64
conversations = [
    {"conversation": json.dumps([{"role": "user", "content": f"Count from {start} to {end}: "}])}
    for start in range(100)
    for end in range(start, 100)
]
print(f"dataset size {len(conversations)}")
override_num_blocks = (len(conversations) + batch_size - 1) // batch_size
ray.init()
concurrency = int(ray.available_resources().get("GPU"))
ds = ray.data.from_items(conversations, override_num_blocks=override_num_blocks)
ds = ds.map_batches(
    Generator, fn_constructor_kwargs=dict(model_id=model_id), batch_size=batch_size, concurrency=concurrency, num_gpus=1
)
df = ds.to_pandas()
df.to_csv("gen.csv")
print(df.head(100))
