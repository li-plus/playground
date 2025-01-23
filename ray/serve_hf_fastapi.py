"""
ray serve: https://docs.ray.io/en/latest/serve/index.html
"""

from __future__ import annotations

import os

import torch
from fastapi import FastAPI
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from ray import serve
from ray.runtime_env import RuntimeEnv

Conversation = list[dict[str, str]]
Conversations = list[Conversation]


app = FastAPI()


@serve.deployment(
    num_replicas=1, ray_actor_options={"num_gpus": 1, "runtime_env": RuntimeEnv(env_vars=dict(os.environ))}
)
@serve.ingress(app)
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

    @app.post("/")
    def generate(self, conversations: Conversations) -> Conversations:
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
        return conversations


app = Generator.bind(model_id="Qwen/Qwen2.5-0.5B-Instruct")

if __name__ == "__main__":
    serve.shutdown()
    serve.run(app)
