# Kernel-driven progress bar in Numba

![Screenshot of progress bar](progress.png)


## What is it?

It's a progress bar whose completion is controlled from a running CUDA kernel
compiled with [https://numba.pydata.org](Numba).


## How do I run it?

```
python kernel_progress.py
```


## What do I need to run it?

* The latest Numba master branch
* A Compute Capability 6.0 or greater CUDA GPU (e.g. GTX 1xxx, RTX 2/3xxx,
  Pascal / Volta / Turing / Ampere Quadro GPUs).
* Linux ([Managed
  Memory](https://numba.readthedocs.io/en/latest/cuda-reference/memory.html#numba.cuda.managed_array)
  in Numba is experimental on Windows, and this use of it seems to crash it).
* ... Maybe other things I forgot to mention.


## Will this work?

Maybe...

* I have not thought hard about synchronization issues between the host and
  device,
* I have not tested this extensively,
* YMMV!


## Should I use this in my program?

Maybe not...

* If you're going to, you should probably read and understand [Section M.2.2 of
  the CUDA Programming
  Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#um-coherency-hd),
  which discusses coherency and concurrency of managed memory.
* You should also examine whether this technique has a performance impact,
* and think about whether breaking a long-running kernel launch into multiple
  shorter kernels would provide a better way of synchronizing progress with the
  host.
