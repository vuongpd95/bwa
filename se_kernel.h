/*
 * se_kernel.h
 */

#ifndef SE_KERNEL_H_
#define SE_KERNEL_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>

#include "bwamem.h"
#include "utils.h"

#define gpuErrchk(ans) { \
	gpuAssert((ans), __FILE__, __LINE__); \
}

#define MAX_BAND_TRY  2
#define THREAD_LIMIT_PER_BLOCK 128
#define WARP_SIZE 32
#define ONE_MBYTE (1024*1024)
#define FIXED_HEAP 1024
#define LEN_SEQ 79
#define N 4096

#endif /* SE_KERNEL_H_ */
