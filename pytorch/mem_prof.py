"""
Reference: https://pytorch.org/blog/understanding-gpu-memory-1/
"""

import torch
from torchvision import models


# Simple Resnet50 example to demonstrate how to capture memory visuals.
def run_resnet50(num_iters=5, device="cuda:0"):
    model = models.resnet50().to(device=device)
    inputs = torch.randn(1, 3, 224, 224, device=device)
    labels = torch.rand_like(model(inputs))
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Start recording memory snapshot history
    torch.cuda.memory._record_memory_history(max_entries=100000)

    for _ in range(num_iters):
        pred = model(inputs)
        loss_fn(pred, labels).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    # Create the memory snapshot file
    torch.cuda.memory._dump_snapshot("mem_prof_log")

    # Stop recording memory snapshot history
    torch.cuda.memory._record_memory_history(enabled=None)


if __name__ == "__main__":
    # Run the resnet50 model
    run_resnet50()
