# Profiler Timeline

## PyTorch DDP

Traced against https://github.com/pytorch/examples/blob/main/imagenet/main.py on 8x V100 for 4 training steps.

```sh
python main.py -a resnet50 --dist-url 'tcp://127.0.0.1:23456' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --dummy
```

## PyTorch AMP

AMP Ref: https://github.com/pytorch/vision/blob/main/references/classification/train.py

## DeepSpeed Zero

Traced against [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat) without `overlap_comm`.

```sh
git clone https://github.com/microsoft/DeepSpeedExamples.git
cd DeepSpeedExamples/applications/DeepSpeed-Chat/training/step1_supervised_finetuning
```

Zero 1

```sh
bash training_scripts/opt/single_node/run_1.3b.sh ./output 1
```

Zero 2

```sh
bash training_scripts/opt/single_node/run_1.3b.sh ./output 2
```

Zero 3

```sh
bash training_scripts/opt/single_node/run_1.3b.sh ./output 3
```
