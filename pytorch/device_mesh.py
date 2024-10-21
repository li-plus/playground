# https://pytorch.org/tutorials/recipes/distributed_device_mesh.html

from torch.distributed.device_mesh import init_device_mesh

mesh_3d = init_device_mesh("cuda", (2, 2, 2), mesh_dim_names=("replicate", "shard", "tp"))

# Users can slice child meshes from the parent mesh.
hsdp_mesh = mesh_3d["replicate", "shard"]
tp_mesh = mesh_3d["tp"]

# Users can access the underlying process group thru `get_group` API.
replicate_group = hsdp_mesh["replicate"].get_group()
shard_group = hsdp_mesh["shard"].get_group()
tp_group = tp_mesh.get_group()

rank = mesh_3d.get_rank()
print(f"[{rank=}] replicate_rank={replicate_group.rank()} shard_rank={shard_group.rank()} tp_rank={tp_group.rank()}")
