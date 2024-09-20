import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

# https://huggingface.co/deepseek-ai
# model_id = 'deepseek-ai/deepseek-coder-6.7b-instruct'
# model_id = 'deepseek-ai/deepseek-coder-7b-instruct-v1.5'
model_id = "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct"


tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side="left")
model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
    model_id,
    attn_implementation="flash_attention_2",
    use_cache=True,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
).eval()


messages = [
    {
        "role": "user",
        "content": """You are an expert in PyTorch and CUDA programming. Please write a custom CUDA PyTorch extension to accelerate the below `model` function.

```python
import torch
import torch.nn.functional as F


def model(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, x.shape[-1:], weight, bias)


x = torch.randn(4096, 1024, device="cuda")
weight = torch.randn(1024, device="cuda")
bias = torch.randn(1024, device="cuda")
model_args = (x, weight, bias)
```""",
    }
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    padding=True,
    return_dict=True,
).to("cuda")
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]
output = model.generate(
    input_ids=input_ids, attention_mask=attention_mask, max_length=2048, eos_token_id=tokenizer.eos_token_id
)
output = output
output_texts = tokenizer.batch_decode(output[:, input_ids.shape[1] :], skip_special_tokens=True)
for text in output_texts:
    print("=" * 100)
    print(text)
