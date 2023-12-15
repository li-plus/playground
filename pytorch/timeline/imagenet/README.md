# Profiler Timeline

## ImageNet

Traced using https://github.com/pytorch/examples/blob/main/imagenet/main.py on 8x V100 for 4 training steps.

```sh
python main.py -a resnet50 --dist-url 'tcp://127.0.0.1:23456' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0 --dummy
```
