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

typedef struct {
	int n, m, first, rid;
	uint32_t w:29, kept:2, is_alt:1;
	float frac_rep;
	int64_t pos;
} flat_mem_chain_t;

typedef struct {
	size_t n, m;
} flat_mem_chain_v;

#endif /* SE_KERNEL_H_ */
