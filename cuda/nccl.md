# NCCL

nccl tests:

Tested on 8 A100 NVLink GPUs with 600GB/s bi-directional bandwidth (300GB/s TX & 300GB/s RX):
```
./build/sendrecv_perf -b 128M -e 1G -f 2 -g 8
#
#                                                              out-of-place                       in-place          
#       size         count      type   redop    root     time   algbw   busbw #wrong     time   algbw   busbw #wrong
#        (B)    (elements)                               (us)  (GB/s)  (GB/s)            (us)  (GB/s)  (GB/s)  
   134217728      33554432     float     sum      -1   1211.2  110.81  110.81      0   1211.8  110.76  110.76    N/A
   268435456      67108864     float     sum      -1   1423.3  188.60  188.60      0   1420.2  189.01  189.01    N/A
   536870912     134217728     float     sum      -1   2785.4  192.75  192.75      0   2772.2  193.66  193.66    N/A
  1073741824     268435456     float     sum      -1   5503.4  195.10  195.10      0   5490.5  195.56  195.56    N/A
```

Note that 195GB/s is uni-directional, meaning that one GPU can simultaneously send and receive 195GB data within one second.
