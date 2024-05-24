import torch
import torch.nn.functional as F
from torch.utils.benchmark import Timer


def logprobs_of_labels(logits, labels):
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs_labels = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1))
    return logprobs_labels.squeeze(-1)


def logprobs_of_labels_v2(logits, labels):
    logprobs_labels = torch.gather(logits, dim=-1, index=labels.unsqueeze(-1))
    logprobs_labels = logprobs_labels - torch.logsumexp(logits, dim=-1, keepdim=True)
    return logprobs_labels.squeeze(-1)


@torch.compile
def logprobs_of_labels_tt(logits, labels):
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs_labels = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1))
    return logprobs_labels.squeeze(-1)


def perf():
    batch_size, seq_len, vocab_size = 32, 2048, 64000
    logits = torch.randn((batch_size, seq_len, vocab_size), dtype=torch.half, device="cuda")
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.long, device="cuda")

    # correctness
    torch.testing.assert_close(
        logprobs_of_labels(logits[:, :-1], input_ids[:, 1:]),
        logprobs_of_labels_v2(logits[:, :-1], input_ids[:, 1:]),
    )
    torch.testing.assert_close(
        logprobs_of_labels(logits[:, :-1], input_ids[:, 1:]),
        logprobs_of_labels_tt(logits[:, :-1], input_ids[:, 1:]),
    )

    # peak memory test
    torch.cuda.empty_cache()
    logprobs_of_labels(logits[:, :-1], input_ids[:, 1:])
    print(
        f"original allocated: {torch.cuda.memory_allocated() / 1e9:.3f} GB, reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB"
    )

    torch.cuda.empty_cache()
    logprobs_of_labels_v2(logits[:, :-1], input_ids[:, 1:])
    print(
        f"optimized allocated: {torch.cuda.memory_allocated() / 1e9:.3f} GB, reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB"
    )

    torch.cuda.empty_cache()
    logprobs_of_labels_tt(logits[:, :-1], input_ids[:, 1:])
    print(
        f"optimized allocated: {torch.cuda.memory_allocated() / 1e9:.3f} GB, reserved: {torch.cuda.memory_reserved() / 1e9:.3f} GB"
    )

    # speed test
    print(
        Timer(stmt="logprobs_of_labels(logits[:, :-1], input_ids[:, 1:])", globals={**globals(), **locals()}).timeit(
            100
        )
    )
    print(
        Timer(stmt="logprobs_of_labels_v2(logits[:, :-1], input_ids[:, 1:])", globals={**globals(), **locals()}).timeit(
            100
        )
    )
    print(
        Timer(stmt="logprobs_of_labels_tt(logits[:, :-1], input_ids[:, 1:])", globals={**globals(), **locals()}).timeit(
            100
        )
    )


perf()
