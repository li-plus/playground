from contextlib import contextmanager

import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2ForCausalLM

model_id = "Qwen/Qwen2-1.5B-Instruct"

model: Qwen2ForCausalLM = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    use_cache=False,
    low_cpu_mem_usage=True,
    device_map="cuda",
)

prompt_len = 128
resp_len = 64
chosen_resp_len = 32
prompt_ids = torch.arange(prompt_len)

chosen_ids = torch.cat(
    (prompt_ids, torch.arange(2048, 2048 + chosen_resp_len), torch.zeros(resp_len - chosen_resp_len, dtype=torch.long))
)
chosen_attention_mask = torch.cat(
    (
        torch.ones(prompt_len + chosen_resp_len, dtype=torch.long),
        torch.zeros(resp_len - chosen_resp_len, dtype=torch.long),
    )
)

rejected_ids = torch.cat((prompt_ids, torch.arange(4096, 4096 + resp_len)))
rejected_attention_mask = torch.ones(prompt_len + resp_len, dtype=torch.long)

input_ids = torch.stack((chosen_ids, rejected_ids), dim=0)
attention_mask = torch.stack((chosen_attention_mask, rejected_attention_mask), dim=0)

input_ids = input_ids.cuda()
attention_mask = attention_mask.cuda()
labels = torch.where(attention_mask.bool(), input_ids, -100)

# warm up
for _ in range(10):
    loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss
    loss.backward()

# zero grad
for param in model.parameters():
    param.grad = None

# reference
ref_output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
ref_output.loss.backward()
ref_logits = ref_output.logits

ref_grad_wte = model.model.embed_tokens.weight.grad
ref_grad_q = model.model.layers[0].self_attn.q_proj.weight.grad

# zero grad
for param in model.parameters():
    param.grad = None


@contextmanager
def shared_prompt(model: Qwen2ForCausalLM, prompt_len: int):
    old_forward = {}

    def wrap_forward(self: Qwen2DecoderLayer, hidden_states: torch.Tensor, *args, **kwargs):
        bs, seq_len, hidden_size = hidden_states.shape
        if self.self_attn.layer_idx > 0:
            # shared to unshared
            assert bs == 1
            resp_len = (seq_len - prompt_len) // 2
            prompt_hidden_states = hidden_states[0, :prompt_len]
            rejected_hidden_states = hidden_states[0, -resp_len:]
            hidden_states = torch.cat(
                (hidden_states[0, : prompt_len + resp_len], prompt_hidden_states, rejected_hidden_states)
            )
            hidden_states = hidden_states.view(2, prompt_len + resp_len, hidden_size)
        else:
            assert bs == 2

        # TODO: unshare for attention only
        outputs = old_forward[self](hidden_states, *args, **kwargs)

        if self.self_attn.layer_idx < self.self_attn.config.num_hidden_layers - 1:
            # unshared to shared
            hidden_states = outputs[0]
            hidden_states = torch.cat((hidden_states[0], hidden_states[1, prompt_len:])).unsqueeze(0)
            outputs = (hidden_states, *outputs[1:])

        return outputs

    for layer in model.model.layers:
        old_forward[layer] = layer.forward
        layer.forward = wrap_forward.__get__(layer)

    try:
        yield
    finally:
        for layer in model.model.layers:
            layer.forward = old_forward[layer]


# shared prompt
with shared_prompt(model, prompt_len=prompt_len):
    opt_output = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    opt_output.loss.backward()
    opt_logits = opt_output.logits

    opt_grad_wte = model.model.embed_tokens.weight.grad
    opt_grad_q = model.model.layers[0].self_attn.q_proj.weight.grad


torch.testing.assert_close(opt_logits, ref_logits)
torch.testing.assert_close(opt_grad_q, ref_grad_q, rtol=1e-2, atol=1e-4)
torch.testing.assert_close(opt_grad_wte, ref_grad_wte, rtol=1e-2, atol=1e-2)
