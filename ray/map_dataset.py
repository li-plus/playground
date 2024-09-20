"""
ray data api: https://docs.ray.io/en/latest/data/api/dataset.html
map_batches: https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.map_batches.html
"""

from __future__ import annotations

import numpy as np

import ray
import ray.data


def transform(batch: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    batch["id"] = batch["id"] + 1
    return batch


ds = ray.data.from_items([{"id": i} for i in range(100000)])
ds = ds.map_batches(transform, batch_size=4, concurrency=2)
df = ds.to_pandas()
print(df.head(10))
