# Profiler Timeline

## PyTorch DDP

Traced against https://github.com/pytorch/examples/blob/main/imagenet/main.py on 8x V100 for 4 training steps. See https://pytorch.org/docs/master/notes/ddp.html for implementation details.
```sh
python main.py -a resnet50 --dist-url 'tcp://127.0.0.1:23456' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --dummy
```

Note:
* Gradients in a bucket are copied to a contiguous buffer. Only one allreduce kernel is needed for each bucket.

## PyTorch AMP

AMP Ref: https://github.com/pytorch/vision/blob/main/references/classification/train.py

## DeepSpeed Zero

Traced against [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) without `overlap_comm`.
```sh
git clone https://github.com/microsoft/DeepSpeedExamples.git
cd DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning
```

Default DeepSpeed Zero config:
```json
{
    "stage": 1,
    "overlap_comm": true,
    "reduce_scatter": true,
    "contiguous_gradients": true,
}
```

**Zero 1: partition optimizer (12 bytes for each params: 4B fp32 param + 4B 1st momentum + 4B 2nd momentum)**
```sh
bash training_scripts/opt/single_node/run_1.3b.sh ./output 1
```

Initialization:
* FP32 params are flattened, packed, and then evenly partitioned across DP group (`single_partition_of_fp32_groups`). Rank 0 holds the first 1/n flat params, rank1 holds the 1/n to 2/n params, and so on.
* Set local optimizer to have flat params of its own partition (`param_group['params'] = [self.single_partition_of_fp32_groups[i]]`)
* In `initialize_optimizer_states`, call `optimizer.step()` so that Adam states (1st & 2nd momentum) are also partitioned.
* FP16 params are flattened but un-partitioned (`bit16_groups_flat`)
* Zero1 forces `overlap_comm=False`, `contiguous_gradients=False`. 
* FP16 uses dynamic loss scaler. BF16 does not use loss scaler.

Forward: local forward.

Backward: backward the entire graph first, and then allreduce all gradients. No overlap of backward & gradient allreduce. See `reduce_gradients`.

Optimize:
* Update local partition of params.
* Allgather the updated model param partitions across dp group.

**Zero 2: partition gradients (2 bytes for each params)**
```sh
bash training_scripts/opt/single_node/run_1.3b.sh ./output 2
```

Initialization:
* Similar to zero 1, but with `partition_gradients=True`.
* Gradient hooks are registered (`register_hook`) to launch allreduce once gradient bucket is ready. May overlap gradient allreduce and backward.

Forward: same as zero1.

Backward:
* Difference with Zero1: [[code]](https://github.com/microsoft/DeepSpeed/blob/c37fe9cbfb8bc10c8dd6ccd8cac9b34ded218990/deepspeed/runtime/zero/stage_1_and_2.py#L1353-L1367)
* Backward to compute local gradients, which are allreduced once their bucket is ready. Then, if a param lies in the local partition, its allreduced gradient is copied to the local gradients buffer (`copy_grads_in_partition`). Otherwise, its gradients are cleared (`clear_grad_attribute`). Note that Zero 1 does not clear any gradient.

Optimize: same as zero1.

**Zero 3: partition model params**
```sh
bash training_scripts/opt/single_node/run_1.3b.sh ./output 3
```

Initialization:
* Each FP16 model param (2B/param) is partitioned across dp group (`_partition_param`) into `param.ds_tensor`. The original `param.data` is released (`free_param`).
* Register module forward & backward hooks (`register_forward_hook`, `register_forward_pre_hook`). Make sure model params are allgathered before forward and partitioned after forward.
* For optimizer, `_create_fp32_partitions` creates FP32 param partition (4B/param). `initialize_optimizer_states` uses temporary dummy FP32 gradient (4B/param) to run optimizer step, creating FP32 adam 1st and 2nd momentum vectors (8B/param). `_setup_for_real_optimizer` creates FP32 gradient buffer (4B/param) for accumulation.
* After initialization, exactly 18 bytes for each param are allocated and are evenly partitioned across dp group. During runtime, FP16 gradients (2B/param) are allocated for backward and freed after optimizer step.

Forward:
* Allgather module params and wait till status turns AVAILABLE (`fetch_sub_module`).
* Prefetch next module params up to `stage3_max_live_parameters` numels. Their status becomes INFLIGHT. The async allgather handles are kept in param status.
* Run normal module forward function to transform activation.
* Partition all params (`release_sub_module`) that will be reused beyond `stage3_max_reuse_distance`.

Backward:
* Allgather module params.
* Run backward to get local gradients (un-partitioned) of params and inputs.
* The gradient hook (`reduce_ready_partitions_and_remove_grads`) copies full gradients to a continuous flat buffer (`__add_grad_to_ipg_bucket`) and triggers reduce-scatter for any ready bucket (`__reduce_and_partition_ipg_grads`). Eventually each rank has its own partition of gradients.

Optimize:
* Convert contiguous fp16 gradients to fp32 gradients (`_prepare_fp32_grad_for_sub_group`) and release fp16 gradients.
* Scale fp32 gradients and do gradient clip (`unscale_and_clip_grads`).
* Bind contiguous fp32 gradients to flat fp32 param groups and call optimizer.step (`_optimizer_step`).
* Copy updated flat fp32 params to flat fp16 params and then link separate fp16 params to views of flat params (`_reassign_or_swap_out_partitioned_parameters`).
* Release fp32 gradients (`_release_sub_group`).
* Allgather persisting parameters (`_post_step`).

Reference:
* https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/zero/stage_1_and_2.py
* https://github.com/microsoft/DeepSpeed/blob/master/deepspeed/runtime/zero/stage3.py

## Megatron-LM

* https://github.com/NVIDIA/Megatron-LM
* https://huggingface.co/blog/megatron-training

Download tokenizer:
```sh
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
wget https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
```

Download and pre-process wikitext2 dataset.
```python
from datasets import load_dataset

dataset = load_dataset(path="wikitext", name="wikitext-2-v1", split="train")
dataset.to_json("wikitext2.json", lines=True)
```

```sh
python3 tools/preprocess_data.py \
        --input wikitext2.json \
        --output-prefix wikitext2 \
        --vocab-file gpt2-vocab.json \
        --tokenizer-type GPT2BPETokenizer \
        --merge-file gpt2-merges.txt \
        --append-eod \
        --workers 96
```

Modify the entrypoint `pretrain_gpt_distributed_with_mp.sh`:
```sh
CHECKPOINT_PATH=./output/
VOCAB_FILE=./gpt2-vocab.json
MERGE_FILE=./gpt2-merges.txt
DATA_PATH=wikitext2_text_document
```

Add extra options: `--use-flash-attn --bf16 --use-distributed-optimizer`

Start training (TP=2 PP=2 DP=2)
```sh
bash examples/pretrain_gpt_distributed_with_mp.sh
```

Initialization:
* Model: model weights are sharded across tensor & pipeline parallel group, and are replicated across data parallel group. Layers are evenly sharded across pipeline parallel group. Within each layer, attention weights are sharded by head and mlp weights are sharded by intermediate hidden dim across tensor parallel group. Rank allocation is: (inner) TP-DP-PP (outer).
* Optimizer: optimizer states of each rank are sharded across its data parallel group if distributed optimizer is enabled.

Forward / Backward: controled by scheduler

Gradient Allreduce:
* All gradients are reduce-scattered across data parallel group.
* Layernorm gradients are allreduced across tensor parallel group.
* Word embedding gradients of first and last pp rank are allreduced across embedding group, if wte weight is tied to lm head. Note that embedding is in a separate bucket.

Optimizer:
* Update local param partition with reduce-scattered local gradients.
* Allgather updated model weights across data parallel group.
