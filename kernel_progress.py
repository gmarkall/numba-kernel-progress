from numba import cuda
from time import sleep
import numpy as np
from progress.bar import Bar


# Timing parameters - adjust if things are too slow / fast for your liking and
# / or setup.

# Number of iterations in the kernel
MAX_VAL = 1000000

# How long the kernel sleeps each iteration
KERNEL_SLEEP = 1000

# How long the host sleeps before checking the progress of the kernel
HOST_SLEEP = 0.001


# A kernel that increments a counter. Simply looping and incrementing may be
# too fast, so we use nanosleep to wait a bit at each iteration.

@cuda.jit
def report_progress(progress):
    for i in range(MAX_VAL):
        cuda.atomic.inc(progress, 0, MAX_VAL)
        cuda.nanosleep(KERNEL_SLEEP)


# We use a managed array for data accessible by both the device and host
progress = cuda.managed_array(1, dtype=np.uint64)

# Initialize counter to zero
progress[0] = 0

# Kernel launch, which runs asynchronously with respect to the host
report_progress[1, 1](progress)


# Report progress until completion

bar = Bar('Kernel', max=MAX_VAL)

val = 0

while val < MAX_VAL:
    sleep(0.001)
    val = progress[0]
    bar.goto(val)

print()
