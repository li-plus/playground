import torch
from liger_kernel.transformers import apply_liger_kernel_to_llama
from transformers import AutoModelForCausalLM


def show_memory_usage(message: str):
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    print(f"[{message}]: {allocated=:.2f} GB, {reserved=:.2f} GB")


input_ids = torch.randint(0, 10000, (8, 1024), device="cuda")
attention_mask = torch.ones_like(input_ids)


def run_model():
    torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="cuda",
    )
    show_memory_usage("pre-forward")
    output = model(input_ids=input_ids, attention_mask=attention_mask)
    show_memory_usage("post-forward")
    output.logits.sum().backward()
    show_memory_usage("post-backward")


print("===== native =====")
run_model()

print("===== liger =====")
apply_liger_kernel_to_llama()
run_model()
