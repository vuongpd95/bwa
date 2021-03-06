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
#include "ksw.h"
#include "kvec.h"

#ifdef __GNUC__
#define LIKELY(x) __builtin_expect((x),1)
#else
#define LIKELY(x) (x)
#endif

#define gpuErrchk(ans) { \
	gpuAssert((ans), __FILE__, __LINE__); \
}

#define MAX_BAND_TRY 2
#define WARP 1024
#define ONE_MBYTE (1024*1024)


#endif /* SE_KERNEL_H_ */
