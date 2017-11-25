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

#include "bwamem.h"
#include "bntseq.h"
#include "utils.h"

#define gpuErrchk(ans) { \
	gpuAssert((ans), __FILE__, __LINE__); \
}

#endif /* SE_KERNEL_H_ */
