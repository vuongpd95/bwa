/*********************
 * *********************************************************
* PROGRAM: se_kernel
* PURPOSE: This is a collection of functions which is intened to optimize the 
* 	speed of seed extension step in BWA MEM procedure.
*
*
* NAME: Vuong Pham-Duy.
*		College student.
*       Faculty of Computer Science and Engineering.
*       Ho Chi Minh University of Technology, Viet Nam.
*       vuongpd95@gmail.com
*
* DATE: 5/10/2017
*
******************************************************************************/
#include "se_kernel.h"

extern "C" void cuda_mem_chain2aln(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac, \
		int l_query, const uint8_t *query, const mem_chain_t *c, mem_alnreg_v *av);

/* CUDA support function */
void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "[M::%s] GPUassert: %s %s %d\n", __func__, \
			cudaGetErrorString(code), file, line);
		if (abort) 
			exit(code);
	}
}

void print_mem_info()
{
	size_t free_byte;
	size_t total_byte;
	gpuErrchk(cudaMemGetInfo(&free_byte, &total_byte));

	double free_db = (double)free_byte;
	double total_db = (double)total_byte;
	double used_db = total_db - free_db;

	fprintf(stderr, "[M::%s] GPU memory usage: used = %.2f MB, free = %.2f MB, total = %.2f MB\n", \
		__func__, used_db/ONE_MBYTE, free_db/ONE_MBYTE, total_db/ONE_MBYTE);
}

void cuda_mem_chain2aln(const mem_opt_t *opt, const bntseq_t *bns, const uint8_t *pac, int l_query, \
		const uint8_t *query, const mem_chain_t *c, mem_alnreg_v *av) {

}
