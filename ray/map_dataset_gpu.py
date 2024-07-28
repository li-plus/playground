"""
Stateful transform: https://docs.ray.io/en/latest/data/transforming-data.html#stateful-transforms
"""

from __future__ import annotations
import ray
import ray.data
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel


class TorchPredictor:
    def __init__(self):
        model_name_or_path = 'THUDM/glm-4-9b-chat'
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.half,
            device_map='auto',
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).eval()

    def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        batch_messages = [[{'role': 'user', 'content': f'Count from {s} to {e}: '}] for s, e in zip(batch['start'], batch['end'])]
        inputs = self.tokenizer.apply_chat_template(batch_messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True
        ).to('cuda')

        outputs = self.model.generate(**inputs, do_sample=False)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        output_texts = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_texts = [x.strip() for x in output_texts]
        batch['output'] = output_texts
        return batch


ranges = [(start, end) for start in range(100) for end in range(start, 100)]
print(f'dataset size {len(ranges)}')
ds = ray.data.from_items([{'start': s, 'end': e} for s, e in ranges])
ds = ds.map_batches(TorchPredictor, batch_size=16, concurrency=8, num_gpus=1)
df = ds.to_pandas()
df.to_csv('gen.csv')
print(df.head(100))
