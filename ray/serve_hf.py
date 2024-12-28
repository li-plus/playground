"""
ray serve: https://docs.ray.io/en/latest/serve/getting_started.html
"""

from __future__ import annotations

import torch
from starlette.requests import Request
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

from ray import serve

Conversation = list[dict[str, str]]
Conversations = list[Conversation]

model_id = "Qwen/Qwen2.5-0.5B-Instruct"


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 16, "num_gpus": 1})
class Generator:
    def __init__(self) -> None:
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

    def generate(
        self,
        conversations: Conversations,
    ) -> Conversations:
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

    async def __call__(self, http_request: Request) -> str:
        payload: str = await http_request.json()
        return self.generate(payload)


app = Generator.bind()
